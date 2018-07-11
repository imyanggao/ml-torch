require('nn')
require('nngraph')

local net = {}

function net.declare(lib)
  lib = lib or 'nn'
  assert(lib == 'nn' or lib == 'cunn' or lib == 'cudnn',
         "parameter lib should only be 'nn', or 'cunn', or 'cudnn'")
  require(lib)
  local modules = {}
  if lib == 'cudnn' then
    require('cunn')             -- for some modules that doesn't have cudnn equivalents.
    modules =
      {cudnn.ReLU, cudnn.Tanh, cudnn.Sigmoid,
       cudnn.SpatialConvolution, cudnn.VolumetricConvolution,
       cudnn.SpatialFullConvolution, cudnn.VolumetricFullConvolution,
       cudnn.SpatialMaxPooling, cudnn.VolumetricMaxPooling, 
       cudnn.SpatialAveragePooling, cudnn.VolumetricAveragePooling,
       cudnn.SoftMax, cudnn.SpatialSoftMax,
       cudnn.LogSoftMax, cudnn.SpatialLogSoftMax,
       cudnn.BatchNormalization,
       cudnn.SpatialBatchNormalization,
       cudnn.VolumetricBatchNormalization}
  else
    modules =
      {nn.ReLU, nn.Tanh, nn.Sigmoid,
       nn.SpatialConvolution, nn.VolumetricConvolution,
       nn.SpatialFullConvolution, nn.VolumetricFullConvolution,
       nn.SpatialMaxPooling, nn.VolumetricMaxPooling, 
       nn.SpatialAveragePooling, nn.VolumetricAveragePooling,
       nn.SoftMax, nn.SpatialSoftMax,
       nn.LogSoftMax, nn.SpatialLogSoftMax,
       nn.BatchNormalization,
       nn.SpatialBatchNormalization,
       nn.VolumetricBatchNormalization}
  end
  ReLU, Tanh, Sigmoid,
  Conv2D, Conv3D,
  FConv2D, FConv3D,
  MaxPool2D, MaxPool3D,
  AvgPool2D, AvgPool3D,
  SoftMax, SoftMax2D,
  LogSoftMax, LogSoftMax2D,
  BN, BN2D, BN3D = table.unpack(modules)
end

-- Check if a particular GPU (by gpuid with non-positive# to use CPU) is available with
-- a specified size of GPU memory (by minSz in Gigabyte):
-- If available, use the device and use the fastest available gpu lib;
-- otherwise, an error issues.
-- In multi-gpu training, this function will automatically choose the gpu
-- with the largest free gpu memory as the "main" gpu.
-- An optional RNG seed could be specified to allow reproduce random sequence.  
function net.gpu(gpuid, minSz, seed)
  if #gpuid == 0 or (#gpuid == 1 and gpuid[1] <= 0) then
    -- no check on CPU memory usage, so minSz don't need to be specified
    if seed ~= nil then
      torch.manualSeed(seed)
    end
    net.declare('nn')
    return 0
  else
    local cunn_found, cudnn_found = pcall(require, 'cunn'), pcall(require, 'cudnn')
    assert(cunn_found == true, 'package cunn not found!')
    assert(minSz ~= nil, 'minSz must be specified to use GPU)')
    local freeMem = torch.Tensor(#gpuid)
    for i = 1, #gpuid do
      assert(gpuid[i] <= cutorch.getDeviceCount(),
             'gpuid should be less than the #GPU = ' .. cutorch.getDeviceCount() .. ' !')
      local freeMemBytes, totalMemBytes = cutorch.getMemoryUsage(gpuid[i])
      local freeMemGB, totalMemGB = freeMemBytes/math.pow(2,30), totalMemBytes/math.pow(2,30)
      assert(freeMemGB > minSz,
             'GPU ' .. sys.COLORS.yellow .. gpuid[i] .. sys.COLORS.none .. ' free memory: [' ..
               sys.COLORS.red .. math.ceil(freeMemGB * 10) * 0.1 .. sys.COLORS.none .. 'GB/' ..
               sys.COLORS.green .. math.floor(totalMemGB * 10) * 0.1 ..
               sys.COLORS.none .. 'GB]. NOT enough memories.')
      freeMem[i] = freeMemBytes
    end
    -- sort gpuid according to the free gpu memory size
    -- this would ease the out-of-memory error in multi-gpu training
    local _, index = torch.sort(freeMem, true)
    local originid = utility.tbl.clone(gpuid)
    for i = 1, index:nElement() do
      gpuid[i] = originid[index[i]]
    end
    cutorch.setDevice(gpuid[1])
    if seed ~= nil then
      torch.manualSeed(seed)
      cutorch.manualSeed(seed)
    end
    if cudnn_found == true then
      net.declare('cudnn')
      return 2
    else
      net.declare('cunn')
      return 1
    end
  end
end


-- input: any number of nn.gModule and nn.Module
-- if tbl_net is given, use table.unpack(tbl_net) as input
function net.getParameters(...)
  -- get parameters
  local networks = {...}
  local parameters = {}
  local gradParameters = {}
  for i = 1, #networks do
    local net_params, net_grads = networks[i]:parameters()

    if net_params then
      for _, p in pairs(net_params) do
        parameters[#parameters + 1] = p
      end
      for _, g in pairs(net_grads) do
        gradParameters[#gradParameters + 1] = g
      end
    end
  end

  local function storageInSet(set, storage)
    local storageAndOffset = set[torch.pointer(storage)]
    if storageAndOffset == nil then
      return nil
    end
    local _, offset = table.unpack(storageAndOffset)
    return offset
  end

  -- this function flattens arbitrary lists of parameters,
  -- even complex shared ones
  local function flatten(parameters)
    if not parameters or #parameters == 0 then
      return torch.Tensor()
    end
    local Tensor = parameters[1].new

    local storages = {}
    local nParameters = 0
    for k = 1,#parameters do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        storages[torch.pointer(storage)] = {storage, nParameters}
        nParameters = nParameters + storage:size()
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k = 1,#parameters do
      local storageOffset = storageInSet(storages, parameters[k]:storage())
      parameters[k]:set(flatStorage,
                        storageOffset + parameters[k]:storageOffset(),
                        parameters[k]:size(),
                        parameters[k]:stride())
      parameters[k]:zero()
    end

    local maskParameters=  flatParameters:float():clone()
    local cumSumOfHoles = flatParameters:float():cumsum(1)
    local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
    local flatUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()

    for k = 1,#parameters do
      local offset = cumSumOfHoles[parameters[k]:storageOffset()]
      parameters[k]:set(flatUsedStorage,
                        parameters[k]:storageOffset() - offset,
                        parameters[k]:size(),
                        parameters[k]:stride())
    end

    for _, storageAndOffset in pairs(storages) do
      local k, v = table.unpack(storageAndOffset)
      flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
    end

    if cumSumOfHoles:sum() == 0 then
      flatUsedParameters:copy(flatParameters)
    else
      local counter = 0
      for k = 1,flatParameters:nElement() do
        if maskParameters[k] == 0 then
          counter = counter + 1
          flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
        end
      end
      assert (counter == nUsedParameters)
    end
    return flatUsedParameters
  end

  -- flatten parameters and gradients
  local flatParameters = flatten(parameters)
  local flatGradParameters = flatten(gradParameters)

  -- return new flat vector that contains all discrete parameters
  return flatParameters, flatGradParameters
end


-- arg: 'heuristic', 'xavier', 'xavier_caffe', 'kaiming'
-- input (...): any number of nn.gModule and nn.Module
-- initialization affect inside function, get the return is not necessary
function net.init(arg, ...)
  local networks = {...}

  -- "Efficient backprop"
  -- Yann Lecun, 1998
  local function init_heuristic(fan_in, fan_out)
    return math.sqrt(1/(3*fan_in))
  end
  
  -- "Understanding the difficulty of training deep feedforward neural networks"
  -- Xavier Glorot, 2010
  local function init_xavier(fan_in, fan_out)
    return math.sqrt(2/(fan_in + fan_out))
  end

  -- "Understanding the difficulty of training deep feedforward neural networks"
  -- Xavier Glorot, 2010
  local function init_xavier_caffe(fan_in, fan_out)
    return math.sqrt(1/fan_in)
  end

  -- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
  -- Kaiming He, 2015
  local function init_kaiming(fan_in, fan_out)
    return math.sqrt(4/(fan_in + fan_out))
  end

  local function set(m, method)
    if m.__typename == 'nn.SpatialConvolution' then
      m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
    elseif m.__typename == 'nn.SpatialConvolutionMM' then
      m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
    elseif m.__typename == 'cudnn.SpatialConvolution' then
      m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
    elseif m.__typename == 'nn.LateralConvolution' then
      m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1))
    elseif m.__typename == 'nn.VerticalConvolution' then
      m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
    elseif m.__typename == 'nn.HorizontalConvolution' then
      m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
    elseif m.__typename == 'nn.Linear' then
      m:reset(method(m.weight:size(2), m.weight:size(1)))
    elseif m.__typename == 'nn.TemporalConvolution' then
      m:reset(method(m.weight:size(2), m.weight:size(1)))            
    end
    
    if m.bias then
      m.bias:zero()
    end
  end

  -- choose initialization method
  local method = nil
  if     arg == 'heuristic'    then method = init_heuristic
  elseif arg == 'xavier'       then method = init_xavier
  elseif arg == 'xavier_caffe' then method = init_xavier_caffe
  elseif arg == 'kaiming'      then method = init_kaiming
  else
    assert(false)
  end

  local nnmodule = nil
  for i = 1, #networks do
    if #networks[i]:listModules() == 1 then
      nnmodule = networks[i]
      set(nnmodule, method)
    elseif #networks[i]:listModules() > 1 then
      for j = 1, #networks[i].modules do
        nnmodule = networks[i].modules[j]
        set(nnmodule, method)
      end
    else
      assert(false)
    end
  end
  
  return table.unpack(networks)
end

function net.lrSchemes(optim)
  local nSchemeEpoch = optim.maxEpoch - optim.lrDecayBegEpoch
  local lrRange = optim.lrMax - optim.lrMin
  local stepLin = -lrRange / math.floor(nSchemeEpoch / optim.lrDecayStep)
  local stepExp = math.log(optim.lrMin / optim.lrMax) / math.floor(nSchemeEpoch / optim.lrDecayStep)
  local nEpochCycle, lrRangeCycle, nEpochCycle1 = {}, {}
  if optim.nCycle ~= 1 then
    assert(nSchemeEpoch >= optim.nCycle, 'total #epoch in scheme must be greater than #cycle')
    if optim.cycleLenFactor == 1 then
      nEpochCycle1 = math.floor(nSchemeEpoch / optim.nCycle)
    else
      nEpochCycle1 = math.floor(nSchemeEpoch * (1 - optim.cycleLenFactor)
                                  / (1 - math.pow(optim.cycleLenFactor, optim.nCycle)))
      assert(nEpochCycle1 > 0, 'Supplied nCycle, cycleLenFactor and maxEpoch will lead to less than one epoch in a cycle')
    end
    optim.maxEpoch = 0
    for i = 1, optim.nCycle do
      nEpochCycle[i] = math.floor(nEpochCycle1 * math.pow(optim.cycleLenFactor, i - 1))
      lrRangeCycle[i] = lrRange * math.pow(optim.cycleLrFactor, i - 1)
      optim.maxEpoch = optim.maxEpoch + nEpochCycle[i]
    end
  end
  
  local function linear(iter)
    local iEpoch = math.ceil(iter / optim.nTrainBatch)
    if iEpoch >= optim.lrDecayBegEpoch then
      local iStep = math.floor((iEpoch - optim.lrDecayBegEpoch) / optim.lrDecayStep)
      return stepLin * iStep + optim.lrMax
    else
      return optim.lrMax
    end
  end

  local function exp(iter)
    local iEpoch = math.ceil(iter / optim.nTrainBatch)
    if iEpoch >= optim.lrDecayBegEpoch then
      local iStep = math.floor((iEpoch - optim.lrDecayBegEpoch) / optim.lrDecayStep)
      return optim.lrMax * math.exp(stepExp * iStep)
    else
      return optim.lrMax
    end
  end

  local function clr(iter)
    local epoch = (iter - 1) / optim.nTrainBatch
    local iCycle, cycle = 0
    for i = 1, optim.nCycle do
      if epoch < nEpochCycle[i] then
        cycle = epoch / nEpochCycle[i]
        break
      else
        epoch = epoch - nEpochCycle[i]
        iCycle = iCycle + 1
      end
    end
    iCycle = math.floor(cycle + iCycle) + 1
    cycle = cycle + iCycle - 1
    
    local x = math.abs(2 * cycle - 2 * iCycle + 1)
    return optim.lrMin + lrRangeCycle[iCycle] * (math.max(0, 1 - x))
  end

  local function cosine(iter)
    local epoch = (iter - 1) / optim.nTrainBatch
    local iCycle, cycle = 0
    for i = 1, optim.nCycle do
      if epoch < nEpochCycle[i] then
        cycle = epoch / nEpochCycle[i]
        break
      else
        epoch = epoch - nEpochCycle[i]
        iCycle = iCycle + 1
      end
    end
    iCycle = math.floor(cycle + iCycle) + 1
    return optim.lrMin + 0.5 * lrRangeCycle[iCycle] * (1 + math.cos((cycle - math.floor(cycle)) * math.pi))
  end

  local function none(iter)
    return optim.state[1].learningRate
  end

  if     optim.lrScheme == 'none'   then return none
  elseif optim.lrScheme == 'lin'    then return linear
  elseif optim.lrScheme == 'exp'    then return exp
  elseif optim.lrScheme == 'clr'    then return clr 
  elseif optim.lrScheme == 'cos'    then return cosine
  else
    assert(false)
  end
end

function net.sharedClone(nnmodule, n)
  local params, gradParams
  if nnmodule.parameters then
    params, gradParams = nnmodule:parameters()
    if params ==  nil then
      params = {}
    end
  end

  local paramsNoGrad
  if nnmodule.parametersNoGrad then
    paramsNoGrad = nnmodule:parametersNoGrad()
  end

  local mem = torch.MemoryFile('w'):binary()
  mem:writeObject(nnmodule)

  local clones = {}
  for t = 1, n do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
    local clone = reader:readObject()
    reader:close()

    if nnmodule.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i = 1, #paramsNoGrad do
          cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

-- create a copy of network with new modules and the same tensors
-- shared tensors (e.g. params, etc.) with network, but new metatable
function net.copyNewMetatable(network)
  local copy = {}
  for k, v in pairs(network) do
    if type(v) == 'table' then
      copy[k] = net.copyNewMetatable(v)
    else
      copy[k] = v
    end
  end
  if torch.typename(network) then
    torch.setmetatable(copy, torch.typename(network))
  end
  return copy
end

-- 'conv'  : nn.SpatialConvolution
-- 'fconv' : nn.SpatialFullConvolution, arg is the adj
-- 'dconv' : nn.SpatialDilatedConvolution, arg is the dilation ratio
-- 'pool'  : nn.SpatialMaxPooling and nn.SpatialAveragePooling
-- 'dpool' : nn.SpatialDilatedMaxPooling, arg is the dilation ratio
function net.outputSize(typename, size, kernel, stride, pad, arg)
  if typename == 'conv' then
    return math.floor((size + 2 * pad - kernel) / stride + 1)
  elseif typename == 'fconv' then
    arg = arg or 0
    return (size - 1) * stride - 2 * pad + kernel + arg
  elseif typename == 'dconv' then
    arg = arg or 1
    return math.floor((size + 2 * pad - arg * (kernel - 1) - 1) / stride) + 1
  elseif typename == 'pool' then
    return math.floor((size + 2 * pad - kernel) / stride + 1)
  elseif typename == 'dpool' then
    arg = arg or 1
    return math.floor((size - (arg * (kernel - 1) + 1) + 2 * pad) / stride + 1)
  end
end

function net.outputSize2D(typename, iH, iW, kH, kW, dH, dW, padH, padW, argH, argW)
  local oH = net.outputSize(typename, iH, kH, dH, padH, argH)
  local oW = net.outputSize(typename, iW, kW, dW, padW, argW)
  return oH, oW
end

function net.centerCrop2D(iH, iW, oH, oW)
  local offsetH, offsetW = (iH -oH) / 2, (iW -oW) / 2
  local m = nn.Sequential()
  m:add(nn.Narrow(3, 1 + math.floor(offsetH), iH - 2 * offsetH))
  m:add(nn.Narrow(4, 1 + math.floor(offsetW), iW - 2 * offsetW))
  return m
end

function net.centerCropPad2D(iH, iW, oH, oW)
  local offsetH, offsetW = (oH - iH) / 2, (oW - iW) / 2
  local padL, padT = math.ceil(offsetW), math.ceil(offsetH)
  local padR, padB = (oW - iW) - padL, (oH - iH) - padT
  return nn.SpatialZeroPadding(padL, padR, padT, padB)
end

function net.conv2DBNReLU(nIn, nOut, k, d, pad, iH, iW)
  local m = nn.Sequential()
  m:add(Conv2D(nIn, nOut, k,k, d,d, pad,pad))
  m:add(BN2D(nOut))
  m:add(ReLU(true))
  local oH, oW =  nil, nil
  if iH ~= nil and iW ~= nil then
    oH, oW = net.outputSize2D('conv', iH, iW, k,k, d,d, pad,pad)
  end
  return m, oH, oW
end

function net.graphConv2DBNReLU(nIn, nOut, k, d, pad, iH, iW, str)
  str = str or ''
  local oH, oW =  nil, nil
  if iH ~= nil and iW ~= nil then
    oH, oW = net.outputSize2D('conv', iH, iW, k,k, d,d, pad,pad)
  end
  return function(iNode)
    local conv = Conv2D(nIn, nOut, k,k, d,d, pad,pad)(iNode):annotate({name = 'conv' .. str})
    local bn = BN2D(nOut)(conv):annotate({name = 'bn' .. str})
    local relu = ReLU(true)(bn):annotate({name = 'relu' .. str})
    return relu
  end, oH, oW
end

function net.linearBNReLU(nIn, nOut)
  local m = nn.Sequential()
  m:add(nn.Linear(nIn, nOut))
  m:add(BN(nOut))
  m:add(ReLU(true))
  return m
end

function net.graphLinearBNReLU(nIn, nOut, str)
  str = str or ''
  return function(iNode)
    local linear = nn.Linear(nIn, nOut)(iNode):annotate({name = 'linear' .. str})
    local bn = BN(nOut)(linear):annotate({name = 'bn' .. str})
    local relu = ReLU(true)(bn):annotate({name = 'relu' .. str})
    return relu
  end
end

-- support both update for vanilla LSTM and multidimensional LSTM (MDLSTM)
-- clone is used to avoid sharing parameters accidentally,
-- because each gate should has own parameters.
-- applied module for H are the same for any dim hidden, but for X and H could be different
-- for MDLSTM, prevC, prevH are tables contain cell and hidden states in all dimension
-- MDLSTM differences are n forget gates needs to deal with n-dim previous cell,
-- and each gate takes n-dim hidden into account
function net.updateLSTM(moduleH, moduleX, moduleY, prevC, prevH, x)
  local dim = 1
  assert(torch.type(prevC) == torch.type(prevH), 'prevC and prevH must be the same type')
  if torch.type(prevC) == 'table' then
    assert(#prevC == #prevH, 'error: #prevC ~= #prevH')
    dim = #prevC
  else
    prevC = {prevC}
    prevH = {prevH}
  end
    
  local iI2H, fI2H, oI2H, cI2H, iH2H, fH2H, oH2H, cH2H, iTbl, fTbl, oTbl, cTbl, sumTbl
  local iGate, fGate, oGate, cDecode, nextC, nextH, y
  
  iI2H = moduleX:clone()(x):annotate{name='iI2H'}
  iTbl, iH2H = {iI2H}, {}
  for i = 1, dim do
    iH2H[i] = moduleH:clone()(prevH[i]):annotate{name='iH2H' .. i}
    table.insert(iTbl, iH2H[i])
  end
  iGate = nn.Sigmoid()(nn.CAddTable()(iTbl))

  fI2H = moduleX:clone()(x):annotate{name='fI2H'}
  fTbl, fH2H, fGate = {}, {}, {}
  for i = 1, dim do
    fTbl[i], fH2H[i] = {fI2H}, {}
    for j = 1, dim do
      fH2H[i][j] = moduleH:clone()(prevH[j]):annotate{name='fH2H' .. i .. j}
      table.insert(fTbl[i], fH2H[i][j])
    end
    fGate[i] = nn.Sigmoid()(nn.CAddTable()(fTbl[i]))
  end

  oI2H = moduleX:clone()(x):annotate{name='oI2H'}
  oTbl, oH2H = {oI2H}, {}
  for i = 1, dim do
    oH2H[i] = moduleH:clone()(prevH[i]):annotate{name='oH2H' .. i}
    table.insert(oTbl, oH2H[i])
  end
  oGate = nn.Sigmoid()(nn.CAddTable()(oTbl))

  cI2H = moduleX:clone()(x):annotate{name='cI2H'}
  cTbl, cH2H = {cI2H}, {}
  for i = 1, dim do
    cH2H[i] = moduleH:clone()(prevH[i]):annotate{name='cH2H' .. i}
    table.insert(cTbl, cH2H[i])
  end
  cDecode = nn.Tanh()(nn.CAddTable()(cTbl))

  sumTbl = {nn.CMulTable()({iGate, cDecode})}
  for i = 1, dim do
    table.insert(sumTbl, nn.CMulTable()({fGate[i], prevC[i]}))
  end
  nextC = nn.CAddTable()(sumTbl)
  nextH = nn.CMulTable()({oGate, nn.Tanh()(nextC)})

  if moduleY ~= nil then
    y = moduleY:clone()(nextH)
  else
    y = nil
  end

  return nextC, nextH, y
end

-- input table:  { layer1_dim1_C, layer1_dim2_C,  layer1_dim1_H, layer1_dim2_H,     -- layer 1
--                 layer2_dim1_C, layer2_dim2_C,  layer2_dim1_H, layer2_dim2_H, x}  -- layer 2
-- output table: { layer1_C, layer1_H,     -- layer 1
--                 layer2_C, layer2_H, y}  -- layer 2
function net.LSTM(inputDim, outputDim, hiddenDim, nLayer, lstmDim)
  local inputs, outputs = {}, {}
  for l = 1, nLayer do
    for d = 1, lstmDim do
      table.insert(inputs, nn.Identity()()) -- prevC
    end
    for d = 1, lstmDim do
      table.insert(inputs, nn.Identity()()) -- prevH
    end
  end
  table.insert(inputs, nn.Identity()())     -- x
  
  local prevC, prevH, x, nextC, nextH, y, curInputDim, moduleY
  for l = 1, nLayer do
    local base = 2 * lstmDim * (l - 1)
    prevC = {table.unpack(inputs, base + 1, base + lstmDim)}
    prevH = {table.unpack(inputs, base + lstmDim + 1, base + 2 * lstmDim)}
    if lstmDim == 1 then
      prevC = table.unpack(prevC) -- if one-dimensional lstm, prevC and prevH should not be table
      prevH = table.unpack(prevH)
    end
    if l == 1 then
      x = inputs[#inputs]
      curInputDim = inputDim
    else
      x = outputs[2*(l-1)]
      curInputDim = hiddenDim
    end
    if l < nLayer then
      moduleY = nil
    else
      moduleY = nn.Linear(hiddenDim, outputDim)
    end
    nextC, nextH, y = net.updateLSTM(nn.Linear(hiddenDim, hiddenDim), nn.Linear(curInputDim, hiddenDim), moduleY, prevC, prevH, x)
    table.insert(outputs, nextC)
    table.insert(outputs, nextH)
  end
  table.insert(outputs, y)
  return nn.gModule(inputs, outputs)
end

function net.getPretrainVGGParams(path, tensorType)
  print(sys.COLORS.red .. 'Loading pretrain VGG model parameters: ' .. path)
  local vgg = torch.load(path):type(tensorType)
  local convParams, fcParams = {{},{},{},{},{}}, {}
  convParams[1][1] = vgg:get(1):parameters()
  convParams[1][2] = vgg:get(3):parameters()
  convParams[2][1] = vgg:get(6):parameters()
  convParams[2][2] = vgg:get(8):parameters()
  convParams[3][1] = vgg:get(11):parameters()
  convParams[3][2] = vgg:get(13):parameters()
  convParams[3][3] = vgg:get(15):parameters()
  convParams[4][1] = vgg:get(18):parameters()
  convParams[4][2] = vgg:get(20):parameters()
  convParams[4][3] = vgg:get(22):parameters()
  convParams[5][1] = vgg:get(25):parameters()
  convParams[5][2] = vgg:get(27):parameters()
  convParams[5][3] = vgg:get(29):parameters()
  fcParams[1] = vgg:get(33):parameters()
  fcParams[2] = vgg:get(36):parameters()
  fcParams[3] = vgg:get(39):parameters()
  return convParams, fcParams
end

utility.net = net
