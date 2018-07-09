
local VGG = torch.class('model.VGG')

require('nngraph')

-- imgSz: nChannel * height * width
function VGG:__init(imgSz, nClass, convPlanes, convLayers, pad, fcDims, pretrainPath)
  self:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  self:allFC(fcDims)
  self:createNet()
  self:initNet(pretrainPath)
end

function VGG:initNet(pretrainPath)
  utility.net.init('kaiming', self.network)
  if pretrainPath ~= nil then
    local convParams, fcParams = utility.net.getPretrainVGGParams(pretrainPath)
    if #convParams == #self.convParams then
      for i = 1, #convParams do
        if #convParams[i] == #self.convParams[i] then
          for j = 1, #convParams[i] do
            if self.convParams[i][j][1]:isSameSizeAs(convParams[i][j][1]) then
              for k = 1, 2 do
                self.convParams[i][j][k]:copy(convParams[i][j][k])
              end
            else
              print('conv block ' .. i .. ' layer ' .. j .. ' parameter size in pretrain model does not match')
            end
          end
        else
          print('#conv layer in pretrain model does not match')
        end
      end
    else
      print('#conv block in pretrain model does not match')
    end

    if self.fcParams ~= nil then
      if #fcParams == #self.fcParams then
        for i = 1, #fcParams do
          if self.fcParams[i][1]:isSameSizeAs(fcParams[i][1]) then
            for k = 1, 2 do
              self.fcParams[i][k]:copy(fcParams[i][k])
            end
          else
            print('fc layer ' .. i .. ' parameter size in pretrain model does not match')
          end
        end
      else
        print('#fc layer in pretrain model does not match')
      end
    end
  end
end

function VGG:grpConv(idx)
  local h, w, pad, downscaler, block = self.bridgeH[idx-1], self.bridgeW[idx-1]
  local nIn, nOut = self.convPlanes[idx-1], self.convPlanes[idx]
  self.convParams[idx] = {}
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
    self.convParams[idx][i] = block:get(1):parameters()
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

function VGG:allConv(imgSz, nClass, convPlanes, convLayers, pad)
  if convPlanes ~= nil and convLayers ~= nil then
    assert(#convPlanes == #convLayers, '#convPlanes == #convLayers')
  end
  self.imgSz = imgSz
  self.nClass = nClass
  self.convPlanes = convPlanes or {64, 128, 256, 512, 512}
  self.convLayers = convLayers or {2, 2, 3, 3, 3}
  self.pad = pad or 1
  self.nChannel = imgSz[1]
  self.convH, self.convW = {[0] = imgSz[2]}, {[0] = imgSz[3]}
  self.bridgeH = {[0] = imgSz[2]}
  self.bridgeW = {[0] = imgSz[3]}
  self.convPlanes[0] = self.nChannel
  self.nConv = #self.convLayers
  self.conv, self.bridge = {}, {}
  self.convParams = {}
  for i = 1, self.nConv do
    self.conv[i], self.bridge[i] = self:grpConv(i)
  end
end

function VGG:grpFC(idx)
  local m
  if idx == self.nFC then
    m = nn.Linear(self.fcDims[idx - 1], self.nClass)
    self.fcParams[idx] = m:parameters()
  else
    m = utility.net.linearBNReLU(self.fcDims[idx-1], self.fcDims[idx])
    self.fcParams[idx] = m:get(1):parameters()
  end
  return m
end

function VGG:allFC(fcDims)
  self.fcDims = fcDims or {4096, 4096}
  self.fcDims[0] = self.convPlanes[self.nConv] * self.bridgeH[self.nConv] * self.bridgeW[self.nConv]
  self.nFC = #self.fcDims + 1
  self.fc = {}
  self.fcParams = {}
  for i = 1, self.nFC do
    self.fc[i] = self:grpFC(i)
  end
end

function VGG:createNet()
  local input = nn.Identity()()
  local layer = {[0] = input}
  for i = 1, self.nConv do
    layer[i] = self.conv[i](layer[i-1])
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
