
local UNet = torch.class('modeler.UNet')

require('nngraph')

function UNet:__init(imgSz, nClass, convPlanes, convLayers, pad, pretrainPath)
  self:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  self:allFConv()
  self:allCrop()
  self:createNet()
  self:initNet(pretrainPath)
end

function UNet:initNet(pretrainPath)
  utility.net.init('kaiming', self.network)
end

function UNet:grpConvDown(idx)
  local h, w, pad, downscaler, block = self.bridgeH[idx-1], self.bridgeW[idx-1]
  local nIn, nOut = self.convPlanes[idx-1], self.convPlanes[idx]
  local m = nn.Sequential()
  for i = 1, self.convLayers[idx] do
    if idx == 1 and i == 1 then
      pad = self.pad
    else
      pad = 1
    end
    if i > 1 then
      nIn = nOut
    end
    block, h, w = utility.net.conv2DBNReLU(nIn, nOut, 3, 1, pad, h, w)
    m:add(block)
  end
  self.convH[idx], self.convW[idx] = h, w
  if idx == self.nConv then
    downscaler = AvgPool2D(2,2,2,2)
  else
    downscaler = MaxPool2D(2,2,2,2)
  end
  self.bridgeH[idx], self.bridgeW[idx] = utility.net.outputSize2D('pool', h,w, 2,2, 2,2, 0,0)
  return m, downscaler
end

function UNet:grpConvUp(idx)
  local nIn, nOut = self.convPlanes[self.nConv-idx+1], self.convPlanes[self.nConv-idx]
  local m, block = nn.Sequential()
  for i = 1, self.convLayers[idx] do
    if i > 1 then
      nIn = nOut
    end
    block = utility.net.conv2DBNReLU(nIn, nOut, 3, 1, 1)
    m:add(block)
  end
  return m
end

function UNet:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  if convPlanes ~= nil and convLayers ~= nil then
    assert(#convPlanes == #convLayers, '#convPlanes == #convLayers')
  end
  self.imgSz = imgSz
  self.nClass = nClass
  self.convPlanes = convPlanes or {64, 128, 256, 512, 1024}
  self.convLayers = convLayers or {2, 2, 2, 2, 2}
  self.pad = pad or 1
  self.nChannel = imgSz[1]
  self.convH, self.convW = {}, {}
  self.bridgeH = {[0] = imgSz[2]}
  self.bridgeW = {[0] = imgSz[3]}
  self.convPlanes[0] = self.nChannel
  self.nConv = #self.convLayers
  self.conv, self.bridge = {}, {}
  
  for i = 1, self.nConv do
    self.conv[i], self.bridge[i] = self:grpConvDown(i) 
  end
end

function UNet:grpFConv(idx)
  local m = FConv2D(self.convPlanes[self.nConv-idx+1], self.convPlanes[self.nConv-idx], 2,2, 2,2, 0,0)
  self.fconvH[idx], self.fconvW[idx] =
    utility.net.outputSize2D('fconv', self.fconvH[idx-1], self.fconvW[idx-1], 2,2, 2,2, 0,0)
  return m
end

function UNet:allFConv()
  self.fconvH = {[0] = self.convH[self.nConv]}
  self.fconvW = {[0] = self.convW[self.nConv]}
  self.nFConv = self.nConv - 1
  self.fconv = {}
  self.convUp = {}
  for i = 1, self.nFConv do
    self.fconv[i] = self:grpFConv(i)
    self.convUp[i] = self:grpConvUp(i)
  end
end

function UNet:grpCrop(idx)
  local m = utility.net.centerCropPad2D(self.convH[self.nConv-idx], self.convW[self.nConv-idx],
                                        self.fconvH[idx], self.fconvW[idx])
  return m
end

function UNet:allCrop()
  self.crop = {}
  for i = 1, self.nFConv do
    self.crop[i] = self:grpCrop(i)
  end
end

function UNet:createNet()
  local input = nn.Identity()()
  local gconv = {[0] = input}
  for i = 1, self.nConv do
    if i > 1 then
      gconv[i] = self.conv[i](self.bridge[i-1](gconv[i-1]))
    else
      gconv[i] = self.conv[i](gconv[i-1])
    end
  end
  local gfconv = {[0] = gconv[self.nConv]}
  for i = 1, self.nFConv do
    gfconv[i] = self.convUp[i](nn.JoinTable(1,3)({self.fconv[i](gfconv[i-1]), self.crop[i](gconv[self.nConv-i])}))
  end
  local output = Conv2D(self.convPlanes[1], self.nClass, 1,1, 1,1, 0,0)(gfconv[self.nFConv])
  -- output = FConv2D(self.nClass, self.nClass, 2,2, 2,2, 0,0)(output)
  output = utility.net.centerCropPad2D(self.fconvH[self.nFConv], self.fconvW[self.nFConv],
                                       self.imgSz[2], self.imgSz[3])(output)
  output = LogSoftMax()(nn.View(-1, self.nClass)(nn.Transpose({2,3}, {3,4})(output)))
  self.network = nn.gModule({input}, {output})
end
