
local UNet = torch.class('modeler.UNet')

function UNet:__init(imgSz, nClass, convPlanes, convLayers, pad1, bn, pretrainPath)
  self:makeConv(imgSz, convPlanes, convLayers, pad1, bn)
  self:makeDeconv(nClass)
  self:create()
  self:init(pretrainPath)
end

function UNet:makeConv(imgSz, convPlanes, convLayers, pad1, bn)
  self.imgSz = imgSz
  self.convPlanes = convPlanes or {64, 128, 256, 512, 1024}
  self.convLayers = convLayers or {2, 2, 2, 2, 2}
  self.pad1 = pad1 or 1
  self.bn = bn
  local convKernel, convStride, convPad = 3, 1, 1
  local poolKernel, poolStride, poolPad = 2, 2, 0

  assert(#self.convPlanes == #self.convLayers, 'unet: #convPlanes ~= #convLayers')
  self.nConv = #self.convLayers
  self.convPlanes[0] = self.imgSz[1]
  self.convH, self.convW = {[0] = self.imgSz[2]}, {[0] = self.imgSz[3]}
  self.bridgeH, self.bridgeW = {[0] = self.imgSz[2]}, {[0] = self.imgSz[3]}
  self.conv, self.bridge = {}, {}
  self.convParams, self.convGradParams = {}, {}
  if self.bn == true then
    self.convBNParams, self.convBNGradParams = {}, {}
  end

  for i = 1, self.nConv do
    local h, w, nOut, nIn, p, layer = self.bridgeH[i-1], self.bridgeW[i-1], self.convPlanes[i]
    self.conv[i] = CUDA(nn.Sequential())
    self.convParams[i], self.convGradParams[i] = {}, {}
    if self.bn == true then
      self.convBNParams[i], self.convBNGradParams[i] = {}, {}
    end
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
      layer = CUDA(layer)      
      self.convParams[i][j], self.convGradParams[i][j] = layer:get(1):parameters()
      if self.bn == true then
        self.convBNParams[i][j], self.convBNGradParams[i][j] = layer:get(2):parameters()
      end
      self.conv[i]:add(layer)
    end
    self.convH[i], self.convW[i] = h, w
    self.bridge[i] = CUDA(MaxPool2D(poolKernel, poolKernel, poolStride, poolStride))
    self.bridgeH[i], self.bridgeW[i] =
      utility.net.outputSize2D('pool', self.convH[i], self.convW[i], poolKernel, poolStride, poolPad)
  end
end

function UNet:makeDeconv(nClass)
  self.nClass = nClass
  local deconvKernel, deconvStride, deconvPad = 2, 2, 0
  local convKernel, convStride, convPad = 3, 1, 1
  local convKernel2, convStride2, convPad2 = 1, 1, 0

  self.nDeconv = self.nConv - 1
  self.deconvH, self.deconvW = {[0] = self.convH[self.nConv]}, {[0] = self.convW[self.nConv]}
  self.deconv, self.deconvConfig, self.deconvParams, self.deconvGradParams = {}, {}, {}, {}
  self.cconv, self.cconvParams, self.cconvGradParams, self.crop = {}, {}, {}, {}
  
  -- max fuse lvl is 4
  local kH, kW, sH, sW = deconvKernel, deconvKernel, deconvStride, deconvStride
  for i = 1, self.nDeconv do
    self.deconvConfig[i] = {self.convPlanes[self.nConv-i+1], self.convPlanes[self.nConv-i],
                            kH,kW, sH,sW, deconvPad,deconvPad}
    self.deconv[i] = CUDA(FConv2D(table.unpack(self.deconvConfig[i])))
    self.deconvParams[i], self.deconvGradParams[i] = self.deconv[i]:parameters()
    self.deconvH[i] = utility.net.outputSize('fconv', self.deconvH[i-1], kH, sH, deconvPad)
    self.deconvW[i] = utility.net.outputSize('fconv', self.deconvW[i-1], kW, sW, deconvPad)

    if self.bn == true then
      self.cconv[i] = CUDA(nn.Sequential()
                             :add(utility.net.conv2DBNReLU(self.convPlanes[self.nConv-i+1], self.convPlanes[self.nConv-i],
                                                           convKernel, convStride, convPad))
                             :add(utility.net.conv2DBNReLU(self.convPlanes[self.nConv-i], self.convPlanes[self.nConv-i],
                                                           convKernel, convStride, convPad)))
    else
      self.cconv[i] = CUDA(nn.Sequential()
                             :add(utility.net.conv2DReLU(self.convPlanes[self.nConv-i+1], self.convPlanes[self.nConv-i],
                                                         convKernel, convStride, convPad))
                             :add(utility.net.conv2DReLU(self.convPlanes[self.nConv-i], self.convPlanes[self.nConv-i],
                                                         convKernel, convStride, convPad)))
    end
    -- the usual crop apply to conv
    self.crop[i] = CUDA(utility.net.centerCropPad2D(self.convH[self.nConv-i], self.convW[self.nConv-i],
                                                    self.deconvH[i], self.deconvW[i]))
  end

  -- the last conv map channel to nClass
  self.cconv[self.nDeconv+1] = CUDA(Conv2D(self.convPlanes[1], self.nClass,
                                           convKernel2,convKernel2, convStride2,convStride2, convPad2,convPad2))

  -- one more deconv may need to ensure same output as image size
  -- therefore, we firstly check the size after last fuse.
  -- If the size is not equal to original image size, we apply the last deconv
  if self.deconvH[self.nDeconv] ~= self.convH[0] or self.deconvW[self.nDeconv] ~= self.convW[0] then
    -- with constraint k = 2*s, s > (size+2p) / (in+1)
    sH = math.ceil((self.convH[0] + 2 * deconvPad) / (self.deconvH[self.nDeconv] + 1))
    sW = math.ceil((self.convW[0] + 2 * deconvPad) / (self.deconvW[self.nDeconv] + 1))
    kH = 2 * sH
    kW = 2 * sW
    self.deconvConfig[self.nDeconv+1] = {self.nClass, self.nClass, kH,kW, sH,sW, deconvPad,deconvPad}
    self.deconv[self.nDeconv+1] = CUDA(FConv2D(table.unpack(self.deconvConfig[self.nDeconv+1])))
    self.deconvParams[self.nDeconv+1], self.deconvGradParams[self.nDeconv+1] = self.deconv[self.nDeconv+1]:parameters()
    self.deconvH[self.nDeconv+1] = utility.net.outputSize('fconv', self.deconvH[self.nDeconv], kH, sH, deconvPad)
    self.deconvW[self.nDeconv+1] = utility.net.outputSize('fconv', self.deconvW[self.nDeconv], kW, sW, deconvPad)
    -- the usual crop apply to conv, but the last crop apply to deconv
    self.crop[self.nDeconv+1] =
      CUDA(utility.net.centerCropPad2D(self.deconvH[self.nDeconv+1], self.deconvW[self.nDeconv+1],
                                       self.convH[0], self.convW[0]))
  end
end

function UNet:create()
  local block = {[0] = CUDA(nn.Identity())()}
  for i = 1, self.nConv do
    if i == 1 then
      block[i] = self.conv[i](block[i-1])
    else
      block[i] = self.conv[i](self.bridge[i-1](block[i-1]))
    end
  end
  for i = 1, self.nDeconv do
    block[self.nConv+i] = self.cconv[i](CUDA(nn.JoinTable(1,3))({self.deconv[i](block[self.nConv+i-1]),
                                                                 self.crop[i](block[self.nConv-i])}))
  end
  block[self.nConv+self.nDeconv] = self.cconv[self.nDeconv+1](block[self.nConv+self.nDeconv])
  if #self.deconv > self.nDeconv then
    block[self.nConv+self.nDeconv+1] = self.crop[self.nDeconv+1](self.deconv[self.nDeconv+1](block[self.nConv+self.nDeconv]))
  end
  self.network = CUDA(nn.gModule({block[0]},
                        {LogSoftMax()(nn.View(-1, self.nClass)(nn.Transpose({2,3}, {3,4})(block[#block])))}))
end

function UNet:init(pretrainPath)
  self.pretrainPath = pretrainPath
end
