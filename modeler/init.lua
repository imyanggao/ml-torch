require('nngraph')

modeler = {}

require('modeler.VGG')
require('modeler.GVGG')
require('modeler.FCNVGG')
require('modeler.FCNVGGNOCROP')
require('modeler.FCSLSTMVGG')
require('modeler.UNet')

function modeler.setup(option)
  local useCuda = utility.net.gpu(option.gpu.id, option.gpu.ram, option.gpu.manualSeed)
  utility.net.cuda(useCuda)
  print('\n=> Creating model: ' .. option.model.network)
  local model
  if string.find(option.model.network, 'VGG') then
    local paramsTbl = {option.data.imageSize, option.data.nClass,
                       option.vgg.planes, option.vgg.layers, option.vgg.fc, option.vgg.pad,
                       option.vgg.bn, option.vgg.dropout, option.model.pretrainPath}
    if option.model.network == 'VGG' then
      model = modeler.VGG(table.unpack(paramsTbl))
    elseif option.model.network == 'FCNVGG' then
      model = modeler.FCNVGG(table.unpack(utility.tbl.cat(paramsTbl, option.vgg.fuse, option.vgg.post)))
    elseif option.model.network == 'FCSLSTMVGG' then
      table.insert(paramsTbl, option.vgg.fuse)
      model = modeler.FCSLSTMVGG(table.unpack(utility.tbl.cat(paramsTbl, option.vgg.fc, option.vgg.fuse)))
    end
  elseif string.find(option.model.network, 'UNet') then
    model = modeler.UNet(option.data.imageSize, option.data.nClass,
                         option.unet.planes, option.unet.layers, option.unet.pad, option.model.pretrainPath)
  end

  local criterion = CUDA(nn[option.model.criterion](option.data.hist, true, option.data.ignoreIndex))

  local sNet = utility.checkpoint.load(option)
  if sNet ~= nil then
    modeler.copy(sNet:type(model.network:type()), model.network)
  end

  local _, base, _ = utility.io.splitPath(option.log)
  graph.dot(model.network.fg, base, option.log .. '-model-graph')
  nngraph.setDebug(true)

  -- use more than one GPU
  if #option.gpu.id > 1 then
    local nccl_found = pcall(require, 'nccl')
    nccl_found = false          -- weird observation: use nccl will slow down the GPU sync at the end of train
    model.network = nn.DataParallelTable(1, true, nccl_found):add(model.network, option.gpu.id)
      :threads(
        function()
          require('nngraph')
          if useCuda == 2 then
            require('cudnn')
            -- cudnn.benchmark = true
            -- cudnn.fastest = true
          elseif useCuda == 1 then
            require('cunn')
          end
        end
              )
    model.network.gradInput = nil
  end
  return model, criterion
end

function modeler.copy(s, t)
  local sParams, tParams = s:parameters(), t:parameters()
  assert(#sParams == #tParams, 'modeler.copy: source and target models do not match!')
  for i = 1, #tParams do
    tParams[i]:copy(sParams[i])
  end
  local sBN, tBN = {}, {}
  for i, m in ipairs(s:findModules(BN2D)) do
    sBN[i] = m
  end
  for i, m in ipairs(t:findModules(BN2D)) do
    tBN[i] = m
  end
  assert(#sBN == #tBN, 'modeler.copy: the batch normalization of source and target models do not match!')
  for i = 1, #tBN do
    tBN[i].running_mean:copy(sBN[i].running_mean)
    tBN[i].running_var:copy(sBN[i].running_var)
  end
end

return modeler
