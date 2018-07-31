
local PreFCNVGG, parent = torch.class('modeler.PreFCNVGG', 'modeler.FCNVGG')

function PreFCNVGG:__init(imgSz, nClass, convPlanes, convLayers, fcDims, pad1, bn, dropout, pretrainPath)
  self:makeConv(imgSz, convPlanes, convLayers, pad1, bn)
  self:makeFC(nClass, fcDims, dropout, pretrainPath)
  self:create()
  self:init()
end

function PreFCNVGG:create()
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
  self.network = CUDA(nn.gModule({block[0]},
                        {LogSoftMax()(nn.View(-1, self.nClass)(nn.Transpose({2,3}, {3,4})(block[self.nConv+1])))}))
end

function PreFCNVGG:init()
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
