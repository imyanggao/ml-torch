
local Colormap = torch.class('utility.Colormap')

local BIT
if _VERSION == 'Lua 5.2' then
  BIT = bit32
else
  BIT = bit
end

function Colormap:__init(map, ncolor)
  self.allMaps = {['voc'] = 1}
  map = map or 'voc'
  ncolor = ncolor or 256
  self:set(map, ncolor)
end

function Colormap:reset()
  self:set('voc', 256)
end

local function bitget(x, n)
  if BIT.band(x, BIT.lshift(1, n - 1)) == 0 then
    return 0
  else
    return 1
  end
end

local function cmapVOC(n)
  local cmap = torch.ByteTensor(n, 3)
  for i = 1, n do
    local id = i - 1
    local r, g, b = 0, 0, 0
    for j = 0, 7 do
      r = BIT.bor(r, BIT.lshift(bitget(id, 1), 7 - j))
      g = BIT.bor(g, BIT.lshift(bitget(id, 2), 7 - j))
      b = BIT.bor(b, BIT.lshift(bitget(id, 3), 7 - j))
      id = BIT.rshift(id, 3)
    end
    cmap[i][1] = r
    cmap[i][2] = g
    cmap[i][3] = b
  end
  -- cmap = cmap / 255
  return cmap
end

function Colormap:update()
  if self.map == 'voc' then
    self.cmap = cmapVOC(self.ncolor)
  end
  self:hashmap()
end

-- transfer a n * 3 tensor to n * 1 tensor
local function squeezeTsr(tsr)
  tsr = tsr:double()
  return tsr[{{},1}] + torch.lshift(tsr[{{},2}], 8) + torch.lshift(tsr[{{},3}], 16) + 1000
end

function Colormap:hashmap()
  local key = squeezeTsr(self.cmap)
  self.hash = {}
  for i = 1, key:numel() do
    self.hash[key[i]] = i
  end
end

function Colormap:set(map, ncolor)
  assert(self.allMaps[map], map .. ' is not supported')
  assert(ncolor <= 256 and ncolor >= 1, 'the number of color must be in [1, 256]')
  self.map = map
  self.ncolor = ncolor
  self:update()
end

-- transfer label image to color image with current colormap
function Colormap:l2c(labelImg)
  labelImg = labelImg:view(1, table.unpack(labelImg:size():totable()))
  if labelImg:type() ~= 'torch.ByteTensor' or labelImg:size(1) ~= 1
  or labelImg:min() < 1 or labelImg:max() > self.ncolor then
    error('input label image must be h * w ByteTensor with intensity [1, '
            .. self.ncolor .. '] based on current colormap!')
  end
  -- VOC color label need [1, 21] (background + 20) and 256 (void)
  -- CrossEntropyCriterion need label [1, 22]
  labelImg = labelImg:squeeze():double()
  if self.map == 'voc' then
    labelImg[labelImg:eq(22)] = 256
  end
  local height, width = labelImg:size(1), labelImg:size(2)
  local indices = labelImg:view(labelImg:numel())
  local colorImg = self.cmap:index(1, indices:long()):view(height, width, 3)
  colorImg = colorImg:transpose(2, 3):transpose(1, 2):contiguous():byte()
  return colorImg
end

-- transfer color image to label image
function Colormap:c2l(colorImg)
  if colorImg:type() ~= 'torch.ByteTensor' or colorImg:size(1) ~= 3 then
    error('input color image must be 3 * h * w ByteTensor')
  end
  local height, width = colorImg:size(2), colorImg:size(3)
  local labelImg = squeezeTsr(colorImg:contiguous():view(3, -1):t())
  for i = 1, labelImg:numel() do
    labelImg[i] = self.hash[labelImg[i]]
  end
  labelImg = labelImg:view(1, height, width):byte()
  -- VOC color label would hash to [1, 21] (background + 20) and 256 (void)
  -- CrossEntropyCriterion need label [1, 22]
  if self.map == 'voc' then
    labelImg[labelImg:eq(256)] = 22
  end
  labelImg = labelImg:squeeze(1)
  return labelImg
end
