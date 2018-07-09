
local GVGG = torch.class('model.GVGG')

require('nngraph')

-- imgSz: nChannel * height * width
function GVGG:__init(imgSz, nClass, convPlanes, convLayers, pad, fcDims)
  self:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  self:allFC(fcDims)
  self:createNet()
  self:initNet()
end

function GVGG:initNet()
  utility.net.init('kaiming', self.network)
end

function GVGG:grpConv(idx)
  local h, w, pad, downscaler, str = self.bridgeH[idx-1], self.bridgeW[idx-1], 1
  local nIn, nOut = self.convPlanes[idx-1], self.convPlanes[idx]
  local m = {}
  for i = 1, self.convLayers[idx] do
    if idx == 1 and i == 1 then
      pad = self.pad
    end
    if i > 1 then
      nIn = nOut
    end
    str = idx .. '_' .. i
    m[i], h, w = utility.net.graphConv2DBNReLU(nIn, nOut, 3, 1, pad, h, w, str)
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

function GVGG:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  if convPlanes ~= nil and convLayers ~= nil then
    assert(#convPlanes == #convLayers, '#convPlanes == #convLayers')
  end
  self.imgSz = imgSz
  self.nClass = nClass
  self.convPlanes = convPlanes or {64, 128, 256, 512, 512}
  self.convLayers = convLayers or {2, 2, 3, 3, 3}
  self.pad = pad or 1
  self.nChannel = imgSz[1]
  self.convH, self.convW = {}, {}
  self.bridgeH = {[0] = imgSz[2]}
  self.bridgeW = {[0] = imgSz[3]}
  self.convPlanes[0] = self.nChannel
  self.nConv = #self.convLayers
  self.conv, self.bridge = {}, {}
  
  for i = 1, self.nConv do
    self.conv[i], self.bridge[i] = self:grpConv(i) 
  end
end

function GVGG:grpFC(idx)
  local m
  if idx == self.nFC then
    m = nn.Linear(self.fcDims[idx - 1], self.nClass)
  else
    m = utility.net.linearBNReLU(self.fcDims[idx-1], self.fcDims[idx])
  end
  return m
end

function GVGG:allFC(fcDims)
  self.fcDims = fcDims or {4096, 4096}
  self.fcDims[0] = self.convPlanes[self.nConv] * self.bridgeH[self.nConv] * self.bridgeW[self.nConv]
  self.nFC = #self.fcDims + 1
  self.fc = {}
  for i = 1, self.nFC do
    self.fc[i] = self:grpFC(i)
  end
end

function GVGG:createNet()
  local input = nn.Identity()()
  local layer = {[0] = input}
  for i = 1, self.nConv do
    for j = 1, #self.conv[i] do
      if j == 1 then
        layer[i] = self.conv[i][j](layer[i-1])
      else
        layer[i] = self.conv[i][j](layer[i])
      end
    end
    layer[i] = self.bridge[i](layer[i])
  end
  layer[self.nConv + 1] = nn.View(-1):setNumInputDims(3)(layer[self.nConv])
  -- layer[self.nConv + 1] = nn.Dropout(0.5)(layer[self.nConv + 1])
  for i = 1, self.nFC do
    layer[self.nConv + 1 + i] = self.fc[i](layer[self.nConv + i])
    -- layer[self.nConv + 1 + i] = nn.Dropout(0.5)(layer[self.nConv + 1 + i])
  end
  local output = LogSoftMax()(layer[self.nConv + 1 + self.nFC])
  self.network = nn.gModule({input}, {output})
end
