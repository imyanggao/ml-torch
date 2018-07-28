
local FCSLSTMVGG, parent = torch.class('modeler.FCSLSTMVGG', 'modeler.FCNVGG')

require('nngraph')

-- please notice there is no fc layers in fcn and all fc layers transfer to conv layers
-- fc naming here is just for convention
function FCSLSTMVGG:__init(imgSz, nClass, convPlanes, convLayers, pad, fcDims, fuseLvl, pretrainPath)
  self:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  self:allHidden()
  self:allFC(fcDims)
  self:allFConv(fuseLvl)
  self:allCropConv()
  self:createNet(fuseLvl)
  -- self:backend(fuseLvl)
  self:initNet(pretrainPath)
end

function FCSLSTMVGG:grpHidden(idx)
  self.CHSz[idx] = {self.convPlanes[idx-1], self.convH[idx-1], self.convW[idx-1]} --change on May 1st from idx -> idx -1, bugs?
  return Conv2D(self.convPlanes[idx], self.convPlanes[idx], 1,1, 1,1, 0,0)
end

function FCSLSTMVGG:allHidden()
  self.CHSz = {}
  self.hidden = {}
  for i = 1, self.nConv do
    self.hidden[i] = self:grpHidden(i)
  end
end

-- function FCSLSTMVGG:grpFC(idx)
--   local h, w, k, m = self.fcH[idx-1], self.fcW[idx-1], 1
--   -- 1st kernel size must be 7 to reduce the size of dim from (h+198)/32 to (h+6)/32
--   if idx == 1 then
--     k = 7
--   end
--   if idx == self.nFC then
--     m = Conv2D(self.fcDims[idx-1], self.nClass, k,k, 1,1, 0,0)
--     h, w = utility.net.outputSize2D('conv', h, w, k,k, 1,1, 0,0)
--   else
--     m, h, w = utility.net.conv2DBNReLU(self.fcDims[idx-1], self.fcDims[idx], k, 1, 0, h, w)
--   end
--   self.fcH[idx], self.fcW[idx] = h, w
--   return m
-- end

-- function FCSLSTMVGG:allFC(fcDims)
--   self.fcDims = fcDims or {4096, 4096}
--   self.fcDims[0] = self.convPlanes[self.nConv]
--   self.fcH = {[0] = self.bridgeH[self.nConv]}
--   self.fcW = {[0] = self.bridgeW[self.nConv]}
--   self.nFC = #self.fcDims + 1
--   self.fc = {}
--   for i = 1, self.nFC do
--     self.fc[i] = self:grpFC(i)
--   end
-- end

-- function FCSLSTMVGG:grpFConv(idx)
--   local k, d, m = 4, 2
--   if idx == self.nFConv then
--     k = 64 / math.pow(2, self.nFConv-1)
--     d = k / 2
--   end
--   m = FConv2D(self.nClass, self.nClass, k,k, d,d, 0,0)
--   self.fconvH[idx], self.fconvW[idx] =
--     utility.net.outputSize2D('fconv', self.fconvH[idx-1], self.fconvW[idx-1], k,k, d,d, 0,0)
--   return m
-- end

-- function FCSLSTMVGG:allFConv(fuseLvl)
--   self.fuseLvl = fuseLvl
--   self.fconvH = {[0] = self.fcH[self.nFC]}
--   self.fconvW = {[0] = self.fcW[self.nFC]}
--   self.nFConv = fuseLvl + 1
--   self.fconv = {}
--   for i = 1, self.nFConv do
--     self.fconv[i] = self:grpFConv(i)
--   end
-- end

-- function FCSLSTMVGG:grpCropConv(idx)
--   local m
--   if idx == self.nFConv then
--     m = utility.net.centerCrop2D(self.fconvH[idx], self.fconvW[idx], self.imgSz[2], self.imgSz[3])
--   else
--     local i = self.nConv - idx
--     local conv = Conv2D(self.convPlanes[i], self.nClass, 1,1, 1,1, 0,0)
--     local h, w = utility.net.outputSize2D('conv', self.bridgeH[i], self.bridgeW[i], 1,1, 1,1, 0,0)
--     local crop = utility.net.centerCrop2D(h, w, self.fconvH[idx], self.fconvW[idx])
--     m = nn.Sequential():add(conv):add(crop)
--   end
--   return m
-- end

-- function FCSLSTMVGG:allCropConv()
--   self.crop = {}
--   for i = 1, self.nFConv do
--     self.crop[i] = self:grpCropConv(i)
--   end
-- end

function FCSLSTMVGG:createNet()
  local prevC, prevH, nextC, nextH, x = {}, {}, {}, {}, {[0] = nn.Identity()()}
  for i = 1, self.nConv do
    prevC[i] = nn.Identity()()
    prevH[i] = nn.Identity()()
    nextC[i], nextH[i] = utility.net.updateLSTM(self.hidden[i], self.conv[i], nil, prevC[i], prevH[i], x[i-1])
    x[i] = self.bridge[i](nextH[i])
  end
  local gfc = {[0] = x[self.nConv]}
  for i = 1, self.nFC do
    gfc[i] = self.fc[i](gfc[i-1])
  end
  local gfconv = {[0] = gfc[self.nFC]}
  -- local gfconv = {[0] = nextH[self.nConv]}
  for i = 1, self.nFConv do
    if i == self.nFConv then
      gfconv[i] = self.crop[i](self.fconv[i](gfconv[i-1]))
    else
      gfconv[i] = nn.CAddTable()({self.fconv[i](gfconv[i-1]), self.crop[i](x[self.nConv-i])})
    end
  end
  local output = LogSoftMax()(nn.View(-1, self.nClass)(nn.Transpose({2,3}, {3,4})(gfconv[self.nFConv])))
  local inputs, outputs = {}, {}
  for i = 1, self.nConv do
    table.insert(inputs, prevC[i])
    table.insert(inputs, prevH[i])
    table.insert(outputs, nextC[i])
    table.insert(outputs, nextH[i])
  end
  table.insert(inputs, x[0])
  table.insert(outputs, output)
  self.network = nn.gModule(inputs, outputs)
end
