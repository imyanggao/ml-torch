
local FCNVGG, parent = torch.class('model.FCNVGG', 'model.VGG')

require('nngraph')

-- please notice there is no fc layers in fcn and all fc layers transfer to conv layers
-- fc naming here is just for convention
function FCNVGG:__init(imgSz, nClass, convPlanes, convLayers, pad, fcDims, fuseLvl, pretrainPath)
  self:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  self:allFC(fcDims)
  self:allFConv(fuseLvl)
  self:allCropConv()
  self:createNet(fuseLvl)
  -- self:backend(fuseLvl)
  self:initNet(pretrainPath)
end

function FCNVGG:grpFC(idx)
  local h, w, k, m = self.fcH[idx-1], self.fcW[idx-1], 1
  -- 1st kernel size must be 7 to reduce the size of dim from (h+198)/32 to (h+6)/32
  if idx == 1 then
    -- k = 7
    k = 1
  end
  if idx == self.nFC then
    m = Conv2D(self.fcDims[idx-1], self.nClass, k,k, 1,1, 0,0) -- deconv1,2,3,5,8,9
    -- m = Conv2D(self.fcDims[idx-1], 2*self.nClass, k,k, 1,1, 0,0) -- deconv4
    -- m = Conv2D(self.fcDims[idx-1], self.convPlanes[self.nConv], k,k, 1,1, 0,0) -- deconv6, deconv7
    h, w = utility.net.outputSize2D('conv', h, w, k,k, 1,1, 0,0)
  else
    m, h, w = utility.net.conv2DBNReLU(self.fcDims[idx-1], self.fcDims[idx], k, 1, 0, h, w)
  end
  self.fcH[idx], self.fcW[idx] = h, w
  return m
end

function FCNVGG:allFC(fcDims)
  self.fcDims = fcDims or {4096, 4096}
  self.fcDims[0] = self.convPlanes[self.nConv]
  self.fcH = {[0] = self.bridgeH[self.nConv]}
  self.fcW = {[0] = self.bridgeW[self.nConv]}
  self.nFC = #self.fcDims + 1
  self.fc = {}
  for i = 1, self.nFC do
    self.fc[i] = self:grpFC(i)
  end
end

function FCNVGG:grpFConv(idx)
  -- local k, d, p, m = 4, 2, 0
  -- if idx == self.nFConv then
  --   k = 64 / math.pow(2, self.nFConv-1)
  --   d = k / 2
  --   p = 0
  -- end
  -- local k, d, p, m = 2, 2, 0
  -- if idx == self.nFConv then
  --   k = 64 / math.pow(2, self.nFConv-1)
  --   d = k
  --   p = 0
  -- end
  
  local k, d, p, m = 4, 2, 1
  -- local k, d, p, m = 6, 2, 2    -- deconv5
  -- local nIn, nOut = self.convPlanes[self.nFConv-idx+1], self.convPlanes[self.nFConv-idx] -- deconv6, deconv7
  if idx == self.nFConv then
    k = 64 / math.pow(2, self.nFConv-1)
    d = k / 2
    p = (k - d)/2
    -- nOut = self.nClass          -- deconv6, deconv7
  end
  
  -- if idx > 1 then
  --   nIn = nIn * 2             -- deconv7
  -- end
  m = FConv2D(self.nClass, self.nClass, k,k, d,d, p,p)
  -- m = FConv2D(2*self.nClass, self.nClass, k,k, d,d, p,p) -- deconv4
  -- m = FConv2D(nIn, nOut, k,k, d,d, p,p) -- deconv6, deconv7
  self.fconvH[idx], self.fconvW[idx] =
    utility.net.outputSize2D('fconv', self.fconvH[idx-1], self.fconvW[idx-1], k,k, d,d, p,p)
  return m
end

function FCNVGG:allFConv(fuseLvl)
  self.fuseLvl = fuseLvl
  self.fconvH = {[0] = self.fcH[self.nFC]}
  self.fconvW = {[0] = self.fcW[self.nFC]}
  self.nFConv = fuseLvl + 1
  self.fconv = {}
  for i = 1, self.nFConv do
    self.fconv[i] = self:grpFConv(i)
  end
end

function FCNVGG:grpCropConv(idx)
  local m
  if idx == self.nFConv then
    m = utility.net.centerCrop2D(self.fconvH[idx], self.fconvW[idx], self.imgSz[2], self.imgSz[3])
  else
    local i = self.nConv - idx
    local conv = Conv2D(self.convPlanes[i], self.nClass, 1,1, 1,1, 0,0) -- deconv1-5, deconv8-9
    local h, w = utility.net.outputSize2D('conv', self.bridgeH[i], self.bridgeW[i], 1,1, 1,1, 0,0) -- deconv1-5, deconv8-9
    -- local conv = nn.Identity() -- deconv6, deconv7
    -- local h, w = self.bridgeH[i], self.bridgeW[i] -- deconv6, deconv7
    -- local crop = utility.net.centerCrop2D(h, w, self.fconvH[idx], self.fconvW[idx]) -- not as generalize as pad2d
    
    local crop = utility.net.centerCropPad2D(h, w, self.fconvH[idx], self.fconvW[idx])
    
    -- local crop = nn.SpatialUpSamplingBilinear({oheight = self.fconvH[idx], owidth = self.fconvW[idx]}) -- does not work
    m = nn.Sequential():add(conv):add(crop)
  end
  return m
end

function FCNVGG:allCropConv()
  self.crop = {}
  for i = 1, self.nFConv do
    self.crop[i] = self:grpCropConv(i)
  end
end

function FCNVGG:createNet()
  local input = nn.Identity()()
  local gconv = {[0] = input}
  for i = 1, self.nConv do
    gconv[i] = self.conv[i](gconv[i-1])
    gconv[i] = self.bridge[i](gconv[i])
  end
  local gfc = {[0] = gconv[self.nConv]}
  for i = 1, self.nFC do
    gfc[i] = self.fc[i](gfc[i-1])
  end
  local gfconv = {[0] = gfc[self.nFC]}
  for i = 1, self.nFConv do
    if i == self.nFConv then
      gfconv[i] = self.crop[i](self.fconv[i](gfconv[i-1]))
    else
      gfconv[i] = nn.CAddTable()({self.fconv[i](gfconv[i-1]), self.crop[i](gconv[self.nConv-i])})
      -- gfconv[i] = Conv2D(self.nClass, self.nClass, 3,3, 1,1, 1,1)(gfconv[i]) -- deconv8
      -- gfconv[i] = nn.JoinTable(1,3)({self.fconv[i](gfconv[i-1]), self.crop[i](gconv[self.nConv-i])}) -- deconv4, deconv7
    end
  end
  local output = LogSoftMax()(nn.View(-1, self.nClass)(nn.Transpose({2,3}, {3,4})(gfconv[self.nFConv])))
  self.network = nn.gModule({input}, {output})
end
