
local VGG = torch.class('model.VGG')

function VGG:__init(imgSz, nClass, convPlanes, convLayers, fcDims, pad1, bn, dropout, pretrainPath)
  self:makeConv(imgSz, convPlanes, convLayers, pad1, bn)
  self:makeFC(nClass, fcDims, dropout, pretrainPath)
  self:create()
  self:init()
end

function VGG:makeConv(imgSz, convPlanes, convLayers, pad1, bn)
  self.imgSz = imgSz
  self.convPlanes = convPlanes or {64, 128, 256, 512, 512}
  self.convLayers = convLayers or {2, 2, 3, 3, 3}
  self.pad1 = pad1 or 1
  self.bn = bn
  local convKernel, convStride, convPad = 3, 1, 1
  local poolKernel, poolStride, poolPad = 2, 2, 0

  assert(#self.convPlanes == #self.convLayers, 'vgg: #convPlanes ~= #convLayers')
  self.nConv = #self.convLayers
  self.convPlanes[0] = self.imgSz[1]
  self.convH, self.convW = {[0] = self.imgSz[2]}, {[0] = self.imgSz[3]}
  self.bridgeH, self.bridgeW = {[0] = self.imgSz[2]}, {[0] = self.imgSz[3]}
  self.conv, self.bridge, self.convParams, self.convGradParams = {}, {}, {}, {}

  for i = 1, self.nConv do
    local h, w, nOut, nIn, p, layer = self.bridgeH[i-1], self.bridgeW[i-1], self.convPlanes[i]
    self.conv[i], self.convParams[i], self.convGradParams[i] = nn.Sequential(), {}, {}
    for j = 1, self.convLayers[i] do
      if j == 1 and i == 1 then
        p = self.pad1
      else
        p = convPad
      end
      if j == 1 then
        nIn = self.convPlanes[i-1]
      else
        nIn = self.convPlanes[i]
      end
      if self.bn == true then
        layer, h, w = utility.net.conv2DBNReLU(nIn, nOut, convKernel, convStride, p, h, w)
      else
        layer, h, w = utility.net.conv2DReLU(nIn, nOut, convKernel, convStride, p, h, w)
      end
      self.convParams[i][j], self.convGradParams[i][j] = layer:get(1):parameters()
      self.conv[i]:add(layer)
    end
    self.convH[i], self.convW[i] = h, w
    self.bridge[i] = MaxPool2D(poolKernel, poolKernel, poolStride, poolStride)
    self.bridgeH[i], self.bridgeW[i] =
      utility.net.outputSize2D('pool', self.convH[i], self.convW[i], poolKernel, poolStride, poolPad)
  end
end

function VGG:makeFC(nClass, fcDims, dropout, pretrainPath)
  self.nClass = nClass
  self.fcDims = fcDims or {4096, 4096}
  self.dropout = dropout
  self.pretrainPath = pretrainPath

  self.nFc = 1 + #self.fcDims
  self.fcDims[0] = self.convPlanes[self.nConv] * self.bridgeH[self.nConv] * self.bridgeW[self.nConv]
  self.fc, self.fcParams, self.fcGradParams = {}, {}, {}
  for i = 1, self.nFc do
    if i == self.nFc then
      self.fc[i] = nn.Linear(self.fcDims[i-1], self.nClass)
      self.fcParams[i], self.fcGradParams[i] = self.fc[i]:parameters()
    else
      if self.dropout == true then
        self.fc[i] = utility.net.linearReLUDropout(self.fcDims[i-1], self.fcDims[i])
      else
        if self.bn == true then
          self.fc[i] = utility.net.linearBNReLU(self.fcDims[i-1], self.fcDims[i])
        else
          self.fc[i] = utility.net.linearReLU(self.fcDims[i-1], self.fcDims[i])
        end
      end
      self.fcParams[i], self.fcGradParams[i] = self.fc[i]:get(1):parameters()
    end
  end
end

function VGG:create()
  local block = {[0] = nn.Identity()()}
  for i = 1, self.nConv do
    block[i] = self.bridge[i](self.conv[i](block[i - 1]))
  end
  block[self.nConv + 1] = nn.View(-1):setNumInputDims(3)(block[self.nConv])
  for i = 1, self.nFc do
    block[self.nConv + 1 + i] = self.fc[i](block[self.nConv + i])
  end
  self.network = nn.gModule({block[0]}, {LogSoftMax()(block[self.nConv + 1 + self.nFc])})
end

function VGG:init()
  utility.net.init('kaiming', self.network)
  if self.pretrainPath ~= '' then
    local convParams, fcParams = utility.net.getPretrainVGGParams(self.pretrainPath, self.network:type())
    if #convParams == #self.convParams then
      for i = 1, #convParams do
        if #convParams[i] == #self.convParams[i] then
          for j = 1, #convParams[i] do
            if self.convParams[i][j][1]:isSameSizeAs(convParams[i][j][1]) then
              for k = 1, 2 do
                self.convParams[i][j][k]:copy(convParams[i][j][k])
              end
            else
              print(sys.COLORS.red .. 'conv block ' .. i .. ' layer ' .. j ..
                      ' parameter size in pretrain model does not match')
            end
          end
        else
          print(sys.COLORS.red .. '#conv layer in pretrain model does not match')
        end
      end
    else
      print(sys.COLORS.red .. '#conv block in pretrain model does not match')
    end

    if #fcParams == #self.fcParams then
      for i = 1, #self.fcParams do
        if self.fcParams[i][1]:isSameSizeAs(fcParams[i][1]) then
          for k = 1, 2 do
            self.fcParams[i][k]:copy(fcParams[i][k])
          end
        else
          print(sys.COLORS.red .. 'fc layer ' .. i ..
                  ' parameter size in pretrain model does not match')
        end
      end
    else
      print(sys.COLORS.red .. '#fc layer in pretrain model does not match')
    end

    return convParams, fcParams
  end
end

function VGG:save(paramStr, gradParamStr)
  if paramStr ~= nil then
    for i = 1, self.nConv do
      for j = 1, self.convLayers[i] do
        torch.save(paramsStr .. '-conv' .. i .. '-' .. j .. '-w.txt', self.convParams[i][j][1])
        torch.save(paramsStr .. '-conv' .. i .. '-' .. j .. '-b.txt', self.convParams[i][j][2])
      end
    end
    for i = 1, self.nFc do
      torch.save(paramsStr .. '-fc' .. i .. '-w.txt', self.fcParams[i][1])
      torch.save(paramsStr .. '-fc' .. i .. '-b.txt', self.fcParams[i][2])
    end
  end
  if gradParamStr ~= nil then
    for i = 1, self.nConv do
      for j = 1, self.convLayers[i] do
        torch.save(gradParamsStr .. '-conv' .. i .. '-' .. j .. '-dw.txt', self.convGradParams[i][j][1])
        torch.save(gradParamsStr .. '-conv' .. i .. '-' .. j .. '-db.txt', self.convGradParams[i][j][2])
      end
    end
    for i = 1, self.nFc do
      torch.save(gradParamsStr .. '-fc' .. i .. '-dw.txt', self.fcGradParams[i][1])
      torch.save(gradParamsStr .. '-fc' .. i .. '-db.txt', self.fcGradParams[i][2])
    end
  end
end
