
local FCNVGG, parent = torch.class('modeler.FCNVGG', 'modeler.VGG')

function FCNVGG:__init(imgSz, nClass, convPlanes, convLayers, fcDims, pad1, bn, dropout, pretrainPath, fuse, post)
  self:makeConv(imgSz, convPlanes, convLayers, pad1, bn)
  self:makeFC(nClass, fcDims, dropout, pretrainPath)
  self:makeDeconv(fuse, post)
  self:create()
  self:init()
end

function FCNVGG:makeFC(nClass, fcDims, dropout, pretrainPath)
  self.nClass = nClass
  self.fcDims = fcDims or {4096, 4096}
  self.dropout = dropout
  self.pretrainPath = pretrainPath
  local convKernel, convStride, convPad, pretrainKernel = 1, 1, 0, 7

  self.nFc = 1 + #self.fcDims
  self.fcDims[0] = self.convPlanes[self.nConv]
  self.fcH, self.fcW = {[0] = self.bridgeH[self.nConv]}, {[0] = self.bridgeW[self.nConv]}
  self.fc = {}
  self.fcParams, self.fcGradParams = {}, {}
  if self.bn == true then
    self.fcBNParams, self.fcBNGradParams = {}, {}
  end
  for i = 1, self.nFc do
    local h, w, nIn, nOut = self.fcH[i-1], self.fcW[i-1], self.fcDims[i-1], self.fcDims[i]
    local kH, kW = convKernel, convKernel
    if i == 1 then
      if string.find(self.pretrainPath, 'vgg') then
        kH, kW = pretrainKernel, pretrainKernel
      -- else
      --   -- general way to transfer 1st fc layer to conv layer
      --   -- we don't want to use this to reduce image size to 1*1 in further if don't have a pretrained model
      --   kH, kW = self.fcH[0], self.fcW[0]
      end
    end
    if i == self.nFc then
      nOut = self.nClass
      self.fc[i] = CUDA(Conv2D(nIn, nOut, kH,kW, convStride,convStride, convPad,convPad))
      self.fcParams[i], self.fcGradParams[i] = self.fc[i]:parameters()
    else
      if self.dropout == true then
        self.fc[i] = CUDA(nn.Sequential()
                            :add(Conv2D(nIn, nOut, kH,kW, convStride,convStride, convPad,convPad))
                            :add(ReLU(true))
                            :add(nn.Dropout(0.5)))
      else
        if self.bn == true then
          self.fc[i] = CUDA(nn.Sequential()
                              :add(Conv2D(nIn, nOut, kH,kW, convStride,convStride, convPad,convPad))
                              :add(BN2D(nOut))
                              :add(ReLU(true)))
          self.fcBNParams[i], self.fcBNGradParams[i] = self.fc[i]:get(2):parameters()
        else
          self.fc[i] = CUDA(nn.Sequential()
                              :add(Conv2D(nIn, nOut, kH,kW, convStride,convStride, convPad,convPad))
                              :add(ReLU(true)))
        end
      end
      self.fcParams[i], self.fcGradParams[i] = self.fc[i]:get(1):parameters()
    end
    self.fcH[i] = utility.net.outputSize('conv', h, kH, convStride, convPad)
    self.fcW[i] = utility.net.outputSize('conv', w, kW, convStride, convPad)
  end
end

function FCNVGG:makeDeconv(fuse, post)
  self.fuse = fuse
  self.post = post
  local deconvKernel, deconvStride, deconvPad = 4, 2, 1
  local convKernel, convStride, convPad = 1, 1, 0

  self.deconvH, self.deconvW = {[0] = self.fcH[self.nFc]}, {[0] = self.fcW[self.nFc]}
  self.deconv, self.deconvConfig, self.deconvParams, self.deconvGradParams = {}, {}, {}, {}
  self.cconv, self.crop = {}, {}
  
  -- for post fuse, max fuse lvl is 4; for pre fuse, max fuse lvl could be 5
  local kH, kW, sH, sW = deconvKernel, deconvKernel, deconvStride, deconvStride
  for i = 1, self.fuse do
    self.deconv[i] = CUDA(FConv2D(self.nClass, self.nClass, kH,kW, sH,sW, deconvPad,deconvPad))
    self.deconvConfig[i] = {self.nClass, self.nClass, kH,kW, sH,sW, deconvPad,deconvPad}
    self.deconvParams[i], self.deconvGradParams[i] = self.deconv[i]:parameters()
    self.deconvH[i] = utility.net.outputSize('fconv', self.deconvH[i-1], kH, sH, deconvPad)
    self.deconvW[i] = utility.net.outputSize('fconv', self.deconvW[i-1], kW, sW, deconvPad)
    local j, h, w
    if self.post == true then
      j = self.nConv - i
      h, w = utility.net.outputSize2D('conv', self.bridgeH[j], self.bridgeW[j], convKernel, convStride, convPad)
    else
      j = self.nConv - i + 1
      h, w = utility.net.outputSize2D('conv', self.convH[j], self.convW[j], convKernel, convStride, convPad)
    end
    self.cconv[i] = CUDA(Conv2D(self.convPlanes[j], self.nClass, convKernel,convKernel, convStride,convStride, convPad,convPad))
    -- the usual crop apply to conv
    self.crop[i] = CUDA(utility.net.centerCropPad2D(h, w, self.deconvH[i], self.deconvW[i]))
  end

  -- for both pre/post fuse, one more deconv may need to ensure same output as image size
  -- therefore, we firstly check the size after last fuse.
  -- If the size is not equal to original image size, we apply the last deconv
  -- for both pre/post fuse, the last deconv is special in the sense that
  -- it may not just double size the feature, but need to make the feature size at least the image size
  if self.deconvH[self.fuse] == self.convH[0] and self.deconvW[self.fuse] == self.convW[0] then
    self.nDeconv = self.fuse
  else
    self.nDeconv = self.fuse + 1
    -- with constraint k = 2*s, s > (size+2p) / (in+1)
    sH = math.ceil((self.convH[0] + 2 * deconvPad) / (self.deconvH[self.fuse] + 1))
    sW = math.ceil((self.convW[0] + 2 * deconvPad) / (self.deconvW[self.fuse] + 1))
    kH = 2 * sH
    kW = 2 * sW
    self.deconv[self.nDeconv] = CUDA(FConv2D(self.nClass, self.nClass, kH,kW, sH,sW, deconvPad,deconvPad))
    self.deconvConfig[self.nDeconv] = {self.nClass, self.nClass, kH,kW, sH,sW, deconvPad,deconvPad}
    self.deconvParams[self.nDeconv], self.deconvGradParams[self.nDeconv] = self.deconv[self.nDeconv]:parameters()
    self.deconvH[self.nDeconv] = utility.net.outputSize('fconv', self.deconvH[self.fuse], kH, sH, deconvPad)
    self.deconvW[self.nDeconv] = utility.net.outputSize('fconv', self.deconvW[self.fuse], kW, sW, deconvPad)
    -- the usual crop apply to conv, but the last crop apply to deconv
    self.crop[self.nDeconv] =
      CUDA(utility.net.centerCropPad2D(self.deconvH[self.nDeconv], self.deconvW[self.nDeconv], self.convH[0], self.convW[0]))
  end
end

function FCNVGG:create()
  local block = {[0] = CUDA(nn.Identity())()}
  for i = 1, self.nConv do
    if self.post == true then
      block[i] = self.bridge[i](self.conv[i](block[i-1]))
    else
      if i == 1 then
        block[i] = self.conv[i](block[i-1])
      else
        block[i] = self.conv[i](self.bridge[i-1](block[i-1]))
      end
    end
  end
  if self.post == true then
    block[self.nConv+1] = block[self.nConv]
  else
    block[self.nConv+1] = self.bridge[self.nConv](block[self.nConv])
  end
  for i = 1, self.nFc do
    block[self.nConv+1] = self.fc[i](block[self.nConv+1])
  end
  local j
  for i = 1, self.nDeconv do
    if self.post == true then
      j = self.nConv - i
    else
      j = self.nConv - i + 1
    end
    if i <= self.fuse then
      block[self.nConv+1+i] = CUDA(nn.CAddTable())({self.deconv[i](block[self.nConv+i]), self.crop[i](self.cconv[i](block[j]))})
    else
      block[self.nConv+1+i] = self.crop[i](self.deconv[i](block[self.nConv+i]))
    end
  end
  self.network = CUDA(nn.gModule({block[0]},
                        {LogSoftMax()(nn.View(-1, self.nClass)(nn.Transpose({2,3}, {3,4})(block[self.nConv+1+self.nDeconv])))}))
end

function FCNVGG:init()
  local convParams, fcParams = parent.init(self)

  if fcParams ~= nil then
    if #fcParams ~= #self.fcParams then
      print(sys.COLORS.red .. '#fc layer in pretrain model does not match, but still try to see first few layers')
    end
    for i = 1, #self.fcParams do
      if self.fcParams[i][1]:nElement() == fcParams[i][1]:nElement() then
        print(sys.COLORS.green .. 'fc layer ' .. i .. ' #parameter in pretrain model does match, so reshape and copy')
        for k = 1, 2 do
          self.fcParams[i][k]:copy(fcParams[i][k]:view(self.fcParams[i][k]:size()))
        end
      else
        print(sys.COLORS.red .. 'fc layer ' .. i .. ' #parameter in pretrain model does not match')
        print(table.unpack(utility.tbl.cat('our model size: ', self.fcParams[i][1]:size():totable())))
        print(table.unpack(utility.tbl.cat('pretrain model size: ', fcParams[i][1]:size():totable())))
        local sz
        if self.fcParams[i][1]:size(1) < fcParams[i][1]:size(1) then
          print(sys.COLORS.green .. 'still copy a small part of pretrain fc layer ' .. i)
          sz = self.fcParams[i][1]:size(1)
          self.fcParams[i][1]:copy(fcParams[i][1][{{1,sz},{}}]:view(self.fcParams[i][1]:size()))
          self.fcParams[i][2]:copy(fcParams[i][2][{{1,sz}}]:view(self.fcParams[i][2]:size()))
        else
          print(sys.COLORS.green .. 'although not enough, use all part of pretrain fc layer ' .. i)
          sz = fcParams[i][1]:size(1)
          self.fcParams[i][1][{{1,sz},{},{},{}}]
            :copy(fcParams[i][1][{{1,sz},{}}]:view(self.fcParams[i][1][{{1,sz},{},{},{}}]:size()))
          self.fcParams[i][2][{{1,sz}}]
            :copy(fcParams[i][2][{{1,sz}}]:view(self.fcParams[i][2][{{1,sz}}]:size()))
        end
      end
    end
  end

  -- for i = 1, self.nDeconv do
  --   local weights = utility.net.fconv2DBilinearWeights(table.unpack(self.deconvConfig[i]))
  --   self.deconvParams[i][1]:copy(weights)
  --   self.deconvParams[i][2]:fill(0)
  -- end
end
