# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'


# ---------------------------------------------------------------------------
# GPU tensor cache
# ---------------------------------------------------------------------------

class GPUTensorCache:
    """
    Wraps a CPU-resident tensor of shape (B, T, ...) and moves per-frame
    slices to GPU on demand.  Tracks which frame indices are already on the
    GPU so we never re-upload unnecessarily.

    Usage
    -----
        cache = GPUTensorCache(cpu_tensor, device)
        gpu_slice = cache.get(frame_indices)   # shape (B, len(indices), ...)
        cache.release(frame_indices)           # free GPU memory for those frames
        cache.release_all()                    # free everything
    """

    def __init__(self, cpu_tensor: torch.Tensor, device: torch.device):
        # Master copy always stays on CPU
        self._cpu = cpu_tensor.cpu()
        self._device = device
        # Dict[int -> gpu_tensor of shape (B, 1, ...)]
        self._gpu_frames: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get(self, indices: list[int]) -> torch.Tensor:
        """Return a GPU tensor containing exactly the requested frame indices,
        preserving their order.  Missing frames are uploaded from CPU."""
        self._ensure_on_gpu(indices)
        parts = [self._gpu_frames[i] for i in indices]
        return torch.cat(parts, dim=1)   # (B, len(indices), ...)

    def release(self, indices: list[int]) -> None:
        """Delete GPU copies of the given frame indices."""
        for i in indices:
            if i in self._gpu_frames:
                del self._gpu_frames[i]
        torch.cuda.empty_cache()

    def release_all(self) -> None:
        self._gpu_frames.clear()
        torch.cuda.empty_cache()

    def size(self, dim: int):
        """Delegate .size() queries to the underlying CPU tensor."""
        return self._cpu.size(dim)

    @property
    def shape(self):
        return self._cpu.shape

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _ensure_on_gpu(self, indices: list[int]) -> None:
        missing = [i for i in indices if i not in self._gpu_frames]
        if not missing:
            return
        # Upload missing frames in one contiguous block where possible,
        # falling back to per-index uploads for non-contiguous gaps.
        for i in missing:
            self._gpu_frames[i] = self._cpu[:, i:i+1].to(self._device)


class GPUFlowCache:
    """
    Same idea but for bidirectional flow tensors where the temporal
    dimension is (T-1) rather than T.

    Wraps a (fwd, bwd) tuple of CPU tensors with shape (B, T-1, 2, H, W).
    """

    def __init__(self, flows_bi: tuple[torch.Tensor, torch.Tensor],
                 device: torch.device):
        self._f_cpu = flows_bi[0].cpu()
        self._b_cpu = flows_bi[1].cpu()
        self._device = device
        self._f_gpu: dict[int, torch.Tensor] = {}
        self._b_gpu: dict[int, torch.Tensor] = {}

    # flow index i corresponds to the transition between frame i and i+1
    def get(self, flow_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure(flow_indices)
        f = torch.cat([self._f_gpu[i] for i in flow_indices], dim=1)
        b = torch.cat([self._b_gpu[i] for i in flow_indices], dim=1)
        return f, b

    def release(self, flow_indices: list[int]) -> None:
        for i in flow_indices:
            self._f_gpu.pop(i, None)
            self._b_gpu.pop(i, None)
        torch.cuda.empty_cache()

    def release_all(self) -> None:
        self._f_gpu.clear()
        self._b_gpu.clear()
        torch.cuda.empty_cache()

    def size(self, dim: int):
        return self._f_cpu.size(dim)

    def _ensure(self, indices: list[int]) -> None:
        for i in indices:
            if i not in self._f_gpu:
                self._f_gpu[i] = self._f_cpu[:, i:i+1].to(self._device)
            if i not in self._b_gpu:
                self._b_gpu[i] = self._b_cpu[:, i:i+1].to(self._device)


# ---------------------------------------------------------------------------
# Original helper functions (unchanged)
# ---------------------------------------------------------------------------

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if frames[0].size != process_size:
            frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if out_size != process_size:
            frames = [f.resize(process_size) for f in frames]
    return frames, process_size, out_size


def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec')
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size
    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask


def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []

    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        masks_img = [Image.open(mpath)]
    else:
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))

    for mask_img in masks_img:
        if size is not None and mask_img.size != size:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    masks_dilated = []
    flow_masks = []

    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

    mask[H_start+dilate_h: H_start+imgH-dilate_h,
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame

    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees',
        help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask',
        help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting',
        choices=['video_inpainting', 'video_outpainting'],
        help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0,
        help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2,
        help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--fp16', action='store_true',
        help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args()

    use_half = True if args.fp16 else False
    if device == torch.device('cpu'):
        use_half = False

    frames, fps, size, video_name = read_frame_from_videos(args.video)
    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)

    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size,
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, \
            'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]

    # Build CPU tensors — never moved to GPU as a whole
    frames_t         = to_tensors()(frames).unsqueeze(0) * 2 - 1        # (1,T,3,H,W)
    flow_masks_t     = to_tensors()(flow_masks).unsqueeze(0)             # (1,T,1,H,W)
    masks_dilated_t  = to_tensors()(masks_dilated).unsqueeze(0)          # (1,T,1,H,W)
    # Keep master copies on CPU; caches manage GPU residency
    frames_cache        = GPUTensorCache(frames_t,        device)
    flow_masks_cache    = GPUTensorCache(flow_masks_t,    device)
    masks_dilated_cache = GPUTensorCache(masks_dilated_t, device)

    ##############################################
    # set up RAFT and flow completion model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'raft-things.pth'),
        model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
        model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
        model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames_t.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')

    with torch.no_grad():
        # ---- compute flow ----
        if frames_t.size(-1) <= 640:
            short_clip_len = 12
        elif frames_t.size(-1) <= 720:
            short_clip_len = 8
        elif frames_t.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        # RAFT runs in fp32 regardless; process one short clip at a time so
        # only the active clip lives on GPU.
        gt_flows_f_list, gt_flows_b_list = [], []
        if frames_t.size(1) > short_clip_len:
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    clip_indices = list(range(f, end_f))
                else:
                    clip_indices = list(range(f - 1, end_f))

                clip_gpu = frames_cache.get(clip_indices)
                flows_f, flows_b = fix_raft(clip_gpu, iters=args.raft_iter)
                # Move flow results to CPU immediately to free GPU memory
                gt_flows_f_list.append(flows_f.cpu())
                gt_flows_b_list.append(flows_b.cpu())
                frames_cache.release(clip_indices)
                torch.cuda.empty_cache()
        else:
            all_indices = list(range(video_length))
            clip_gpu = frames_cache.get(all_indices)
            flows_f, flows_b = fix_raft(clip_gpu, iters=args.raft_iter)
            gt_flows_f_list.append(flows_f.cpu())
            gt_flows_b_list.append(flows_b.cpu())
            frames_cache.release(all_indices)
            torch.cuda.empty_cache()

        # Concatenate on CPU; wrap in a cache for on-demand GPU access later
        gt_flows_f_cpu = torch.cat(gt_flows_f_list, dim=1)
        gt_flows_b_cpu = torch.cat(gt_flows_b_list, dim=1)
        gt_flows_cache = GPUFlowCache((gt_flows_f_cpu, gt_flows_b_cpu), device)

        if use_half:
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        # ---- complete flow ----
        # We accumulate completed flows on CPU, then wrap in a cache.
        flow_length = gt_flows_f_cpu.size(1)
        pred_flows_f_list, pred_flows_b_list = [], []
        pad_len = 5

        def _complete_flow_subclip(s_f, e_f, pad_len_s, pad_len_e):
            """Upload one sub-clip worth of gt_flows + flow_masks, run the
            completion network, and return CPU tensors for the trimmed result."""
            flow_indices  = list(range(s_f, e_f))
            # flow_masks has T frames but flows have T-1 edges;
            # completion network needs mask indices s_f .. e_f (inclusive → e_f+1 frames)
            mask_indices  = list(range(s_f, min(flow_length + 1, e_f + 1)))

            gt_f_gpu, gt_b_gpu = gt_flows_cache.get(flow_indices)
            fm_gpu = flow_masks_cache.get(mask_indices)

            if use_half:
                gt_f_gpu = gt_f_gpu.half()
                gt_b_gpu = gt_b_gpu.half()
                fm_gpu   = fm_gpu.half()

            pred_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                (gt_f_gpu, gt_b_gpu), fm_gpu)
            pred_bi_sub = fix_flow_complete.combine_flow(
                (gt_f_gpu, gt_b_gpu), pred_bi_sub, fm_gpu)

            pf = pred_bi_sub[0][:, pad_len_s: e_f - s_f - pad_len_e].cpu()
            pb = pred_bi_sub[1][:, pad_len_s: e_f - s_f - pad_len_e].cpu()

            gt_flows_cache.release(flow_indices)
            flow_masks_cache.release(mask_indices)
            torch.cuda.empty_cache()
            return pf, pb

        if flow_length > args.subvideo_length:
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                pf, pb = _complete_flow_subclip(s_f, e_f, pad_len_s, pad_len_e)
                pred_flows_f_list.append(pf)
                pred_flows_b_list.append(pb)
        else:
            # Single pass — still use the helper to keep code paths consistent
            all_flow_idx = list(range(flow_length))
            all_mask_idx = list(range(flow_length + 1))
            gt_f_gpu, gt_b_gpu = gt_flows_cache.get(all_flow_idx)
            fm_gpu = flow_masks_cache.get(all_mask_idx)

            if use_half:
                gt_f_gpu = gt_f_gpu.half()
                gt_b_gpu = gt_b_gpu.half()
                fm_gpu   = fm_gpu.half()

            pred_bi, _ = fix_flow_complete.forward_bidirect_flow(
                (gt_f_gpu, gt_b_gpu), fm_gpu)
            pred_bi = fix_flow_complete.combine_flow(
                (gt_f_gpu, gt_b_gpu), pred_bi, fm_gpu)

            pred_flows_f_list.append(pred_bi[0].cpu())
            pred_flows_b_list.append(pred_bi[1].cpu())
            gt_flows_cache.release(all_flow_idx)
            flow_masks_cache.release(all_mask_idx)
            torch.cuda.empty_cache()

        pred_flows_f_cpu = torch.cat(pred_flows_f_list, dim=1)
        pred_flows_b_cpu = torch.cat(pred_flows_b_list, dim=1)
        # Wrap completed flows in a cache
        pred_flows_cache = GPUFlowCache((pred_flows_f_cpu, pred_flows_b_cpu), device)

        # ---- image propagation ----
        # masked_frames = frames * (1 - masks_dilated)  — compute on CPU
        masked_frames_cpu = frames_t * (1 - masks_dilated_t)   # stays on CPU
        masked_frames_cache    = GPUTensorCache(masked_frames_cpu,  device)

        subvideo_length_img_prop = min(100, args.subvideo_length)

        # We'll accumulate updated_frames and updated_masks on CPU
        updated_frames_list = []
        updated_masks_list  = []
        pad_len_img = 10

        def _propagate_subclip(s_f, e_f, pad_len_s, pad_len_e):
            """Run img_propagation for one sub-clip and return trimmed CPU tensors."""
            frame_indices = list(range(s_f, e_f))
            # flows span frame transitions s_f..e_f-1 (i.e. e_f-1 edges)
            flow_indices  = list(range(s_f, e_f - 1))

            mf_gpu  = masked_frames_cache.get(frame_indices)
            md_gpu  = masks_dilated_cache.get(frame_indices)
            pf_gpu, pb_gpu = pred_flows_cache.get(flow_indices)

            if use_half:
                mf_gpu  = mf_gpu.half()
                md_gpu  = md_gpu.half()
                pf_gpu  = pf_gpu.half()
                pb_gpu  = pb_gpu.half()

            b, t = md_gpu.size(0), md_gpu.size(1)
            frames_full_gpu = frames_cache.get(frame_indices)
            if use_half:
                frames_full_gpu = frames_full_gpu.half()

            prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                mf_gpu, (pf_gpu, pb_gpu), md_gpu, 'nearest')

            uf_sub = frames_full_gpu * (1 - md_gpu) + \
                     prop_imgs_sub.view(b, t, 3, h, w) * md_gpu
            um_sub = updated_local_masks_sub.view(b, t, 1, h, w)

            uf_out = uf_sub[:, pad_len_s: e_f - s_f - pad_len_e].cpu()
            um_out = um_sub[:, pad_len_s: e_f - s_f - pad_len_e].cpu()

            masked_frames_cache.release(frame_indices)
            masks_dilated_cache.release(frame_indices)
            pred_flows_cache.release(flow_indices)
            frames_cache.release(frame_indices)
            torch.cuda.empty_cache()
            return uf_out, um_out

        if video_length > subvideo_length_img_prop:
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len_img)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len_img)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                uf, um = _propagate_subclip(s_f, e_f, pad_len_s, pad_len_e)
                updated_frames_list.append(uf)
                updated_masks_list.append(um)
        else:
            s_f, e_f = 0, video_length
            frame_indices = list(range(video_length))
            flow_indices  = list(range(video_length - 1))

            mf_gpu  = masked_frames_cache.get(frame_indices)
            md_gpu  = masks_dilated_cache.get(frame_indices)
            pf_gpu, pb_gpu = pred_flows_cache.get(flow_indices)
            frames_full_gpu = frames_cache.get(frame_indices)

            if use_half:
                mf_gpu  = mf_gpu.half()
                md_gpu  = md_gpu.half()
                pf_gpu  = pf_gpu.half()
                pb_gpu  = pb_gpu.half()
                frames_full_gpu = frames_full_gpu.half()

            b, t = md_gpu.size(0), md_gpu.size(1)
            prop_imgs, updated_local_masks = model.img_propagation(
                mf_gpu, (pf_gpu, pb_gpu), md_gpu, 'nearest')
            uf = frames_full_gpu * (1 - md_gpu) + \
                 prop_imgs.view(b, t, 3, h, w) * md_gpu
            um = updated_local_masks.view(b, t, 1, h, w)

            updated_frames_list.append(uf.cpu())
            updated_masks_list.append(um.cpu())

            masked_frames_cache.release(frame_indices)
            masks_dilated_cache.release(frame_indices)
            pred_flows_cache.release(flow_indices)
            frames_cache.release(frame_indices)
            torch.cuda.empty_cache()

        updated_frames_cpu = torch.cat(updated_frames_list, dim=1)   # (1,T,3,H,W) on CPU
        updated_masks_cpu  = torch.cat(updated_masks_list,  dim=1)   # (1,T,1,H,W) on CPU

        # Caches for the transformer stage
        updated_frames_cache = GPUTensorCache(updated_frames_cpu, device)
        updated_masks_cache  = GPUTensorCache(updated_masks_cpu,  device)
        # Re-initialise masks_dilated_cache (was partially released above)
        masks_dilated_cache  = GPUTensorCache(masks_dilated_t, device)
        # pred_flows_cache was partially released; rebuild from the full CPU tensors
        pred_flows_cache = GPUFlowCache((pred_flows_f_cpu, pred_flows_b_cpu), device)

    ori_frames  = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    ref_num = (args.subvideo_length // args.ref_stride) if video_length > args.subvideo_length else -1

    # ---- feature propagation + transformer ----
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        all_ids = neighbor_ids + ref_ids

        # Flow indices for neighbor transitions only
        flow_ids = neighbor_ids[:-1]   # indices of (neighbor_ids[i] → neighbor_ids[i+1]) transitions

        selected_imgs         = updated_frames_cache.get(all_ids)
        selected_masks        = masks_dilated_cache.get(all_ids)
        selected_update_masks = updated_masks_cache.get(all_ids)
        sel_pf, sel_pb        = pred_flows_cache.get(flow_ids)

        if use_half:
            selected_imgs         = selected_imgs.half()
            selected_masks        = selected_masks.half()
            selected_update_masks = selected_update_masks.half()
            sel_pf = sel_pf.half()
            sel_pb = sel_pb.half()

        selected_pred_flows_bi = (sel_pf, sel_pb)

        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi,
                             selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

            # binary_masks: fetch from CPU tensor directly (no GPU needed)
            binary_masks = masks_dilated_t[0, neighbor_ids, :, :, :].permute(
                0, 2, 3, 1).numpy().astype(np.uint8)

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5
                                        + img.astype(np.float32) * 0.5)
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        # Release GPU copies for frames that are no longer a neighbor of any
        # future window.  A frame i will never be needed again once
        # f + neighbor_stride > i + neighbor_stride, i.e. once f > i.
        evict = [i for i in neighbor_ids if i < f]
        updated_frames_cache.release(evict)
        updated_masks_cache.release(evict)
        masks_dilated_cache.release(evict)
        pred_flows_cache.release([i for i in flow_ids if i < f])

        torch.cuda.empty_cache()

    # Tidy up
    updated_frames_cache.release_all()
    updated_masks_cache.release_all()
    masks_dilated_cache.release_all()
    pred_flows_cache.release_all()

    # save each frame
    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)

    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)

    print(f'\nAll results are saved in {save_root}')

    torch.cuda.empty_cache()