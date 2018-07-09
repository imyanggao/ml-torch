require('utility.init')
local matio = require 'matio'
-- local data = matio.load('image.dat')
-- local img_src = data.src
-- local img_tgt = data.tgt
local img_src = image.load('def0.png', 1, 'byte'):squeeze():double()
local img_tgt = image.load('def1.png', 1, 'byte'):squeeze():double()
-- local img_src = image.load('I0.png', 1, 'byte'):squeeze():double()
-- local img_tgt = image.load('I1.png', 1, 'byte'):squeeze():double()
-- local img_src = image.load('0.png', 1, 'byte'):squeeze():double()
-- local img_tgt = image.load('1.png', 1, 'byte'):squeeze():double()
img_src = img_src / img_src:max()
img_tgt = img_tgt / img_tgt:max()
local v0 = torch.zeros(2, table.unpack(img_src:size():totable()))
local deform, dv0, grad_v, Lv0, v_energy, i_energy
local grad_norm = math.huge
local iter = 1
local sigma, lr = 0.03, 1e-4 -- 1e-4 old
local prev_total_energy = math.huge
local geodesic_shooting = utility.GeodesicShooting(img_src:size():totable())
while grad_norm > 1e-4 do
  dv0, deform = geodesic_shooting:shoot(img_src, img_tgt, v0)
  grad_v = v0 + dv0
  grad_norm = torch.norm(grad_v, 2)
  Lv0 = geodesic_shooting:apply(geodesic_shooting.L, v0)
  v_energy = torch.norm(torch.cmul(Lv0, v0), 1)
  i_energy = torch.norm(deform - img_tgt, 2) ^ 2 / (2 * sigma^2)
  if (v_energy + i_energy) < prev_total_energy then
    prev_total_energy = v_energy + i_energy
  else
    break
  end
  print("iter = ", iter, "i_energy = ", i_energy, "v_energy = ", v_energy)
  v0 = v0 - grad_v * lr
  iter = iter + 1
  if iter % 2 == 0 then
    image.save('deform.png', deform)
  end
end
