import sys
sys.path.append(".")

from omegaconf import OmegaConf
config_path = "logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
config = OmegaConf.load(config_path)
import yaml
# print(yaml.dump(OmegaConf.to_container(config)))

from taming.models.cond_transformer import Net2NetTransformer
model = Net2NetTransformer(**config.model.params)

import torch
ckpt_path = "logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)

model.cuda().eval()
torch.set_grad_enabled(False)

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def show_segmentation(s, fname):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
  colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
  colorize = colorize / colorize.sum(axis=2, keepdims=True)
  s = s@colorize
  s = s[...,0,:]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  s.save(fname)

segmentation_path = "data/sflckr_segmentations/norway/25735082181_999927fe5a_b.png"
segmentation = Image.open(segmentation_path)
segmentation = np.array(segmentation)
segmentation = np.eye(182)[segmentation]
segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device) 

show_segmentation(segmentation, "inp1.png") 

c_code, c_indices = model.encode_to_c(segmentation) 
c_code = F.interpolate(c_code, size=(42 * 2, 64 * 2), mode='bilinear')  
print('cind: ', c_indices.shape)
c_indices = c_indices.view(1, 1, 42, 64)  
c_indices = c_indices.float()
print('cind2: ', c_indices.shape)
c_indices = F.interpolate(c_indices, size=(42 * 2, 64 * 2))   
c_indices = c_indices.view(42 * 64 * 2 * 2) 
c_indices = c_indices.long()
# c_indices = F.interpolate(c_indices, size=(42 * 2 * 64 * 2)) 
# c_indices = torch.cat((c_indices,c_indices))
# c_indices = torch.cat((c_indices,c_indices))
print("c_code", c_code.shape, c_code.dtype)
print("c_indices", c_indices.shape, c_indices.dtype)
assert c_code.shape[2]*c_code.shape[3] == c_indices.shape[0]
segmentation_rec = model.cond_stage_model.decode(c_code)
show_segmentation(torch.softmax(segmentation_rec, dim=1), "rec1.png") 

def show_image(s, fname):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  s.save(fname) 

codebook_size = config.model.params.first_stage_config.params.embed_dim
z_indices_shape = c_indices.shape 
# z_indices_shape = torch.Size([c_indices.shape[0] * 4]) 
z_code_shape = c_code.shape 
# z_code_shape = torch.Size([1, 256, 42 * 2, 64 * 2])
z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
x_sample = model.decode_to_img(z_indices, z_code_shape)
show_image(x_sample, "rand1.png")

# from IPython.display import clear_output
import time

idx = z_indices 
print('idx1: ', idx.shape)
idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3]) 
print('idx2: ', idx.shape)

cidx = c_indices
print('cidx1: ', cidx.shape)
cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])
print('cidx2: ', cidx.shape)

temperature = 1.0
top_k = 100
update_every = 50

start_t = time.time()
for i in range(0, z_code_shape[2]-0):
  if i <= 8:
    local_i = i
  elif z_code_shape[2]-i < 8:
    local_i = 16-(z_code_shape[2]-i)
  else:
    local_i = 8
  for j in range(0,z_code_shape[3]-0):
    if j <= 8:
      local_j = j
    elif z_code_shape[3]-j < 8:
      local_j = 16-(z_code_shape[3]-j)
    else:
      local_j = 8

    i_start = i-local_i
    i_end = i_start+16
    j_start = j-local_j
    j_end = j_start+16 

    print('i1, i2, j1, j2: ', i_start, i_end, j_start, j_end) 
    print('i j, local i j: ', i, j, local_i, local_j)
    
    patch = idx[:,i_start:i_end,j_start:j_end] 
    patch = patch.reshape(patch.shape[0],-1) 
    print('patch1: ', patch.shape)
    cpatch = cidx[:, i_start:i_end, j_start:j_end]
    cpatch = cpatch.reshape(cpatch.shape[0], -1) 
    print('cpatch1: ', cpatch.shape)
    patch = torch.cat((cpatch, patch), dim=1) 
    print('patch: ', patch.shape)
    logits,_ = model.transformer(patch[:,:-1]) 
    print('logits1: ', logits.shape)
    logits = logits[:, -256:, :] 
    print('logits2: ', logits.shape)
    logits = logits.reshape(z_code_shape[0],16,16,-1) 
    print('logits3: ', logits.shape)
    logits = logits[:,local_i,local_j,:]
    print('logits4: ', logits.shape)

    logits = logits/temperature

    if top_k is not None:
      logits = model.top_k_logits(logits, top_k) 
    
    print('logits5: ', logits.shape)

    probs = torch.nn.functional.softmax(logits, dim=-1) 
    print('probs: ', probs.shape)
    new_idx = torch.multinomial(probs, num_samples=1) 
    print('new_idx: ', new_idx)
    idx[:,i,j] = new_idx 
    print('LAST: ', idx.shape, c_indices.shape, z_code_shape) 

    # idx = c_indices.view(idx.shape)

    step = i*z_code_shape[3]+j
    if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
      x_sample = model.decode_to_img(idx, z_code_shape)
    #   clear_output()
      print(f"Time: {time.time() - start_t} seconds")
      print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
      show_image(x_sample, f"Sample_Res_6/{i}_{j}.png")