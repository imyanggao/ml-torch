
local opt = {}

function opt.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text('Options:')
  -- torch
  cmd:option('-tensorType',     'FloatTensor',           'set default tensor type')
  -- action options
  cmd:option('-resume',         false,                   'if use -resume, resume from checkpoint (if existed)')
  cmd:option('-retrain',        '',                      'use the model in specified checkpoint to retrain')
  -- record options
  cmd:option('-logStr',         'log',                   'default log str')
  cmd:option('-logDir',         '../log',                'default log dir')
  cmd:option('-example',        'none',                  'example to save, format as train:1,2 test:1,2')
  cmd:option('-exampleEpoch',   1,                       'save example every exampleEpoch')
  cmd:option('-exampleType',    'pred',                  'options: pred|acti|allpred|allacti|all')   
  cmd:option('-checkpointEpoch',50,                      'save checkpoint every checkpointEpoch')
  cmd:option('-pastalog',       8120,                    'pastalog port (0 for not use)')
  -- gpu options
  cmd:option('-manualSeed',     '',                      'when specified, manually set RNG seed')
  cmd:option('-gpuid',          '3',                     'GPU index to use by default')
  cmd:option('-gram',           10,                      'estimation of required gpu memory')
  -- loader options
  cmd:option('-loaderType',     'MTLoader',              'options: MTLoader | SimpleLoader')
  cmd:option('-nThreads',       4,                       'number of threads for data loading')
  cmd:option('-batchSize',      16,                      'batch size')
  -- data options
  cmd:option('-dataset',        'ibsr',                  'options: cifar10 | cifar100 | imagenet | voc | voc-sbd | bric | ibis | ibsr')
  cmd:option('-datasetConfig',  '',                      'when specified, use dataset config file')
  cmd:option('-dataInputType',  'FloatTensor',           'default tensor type for data.input')
  cmd:option('-dataTargetType', 'LongTensor',            'default tensor type for data.target')
  cmd:option('-dataDim',        2,                       'options: 2 | 3 to use as 2d or 3d image (for medical imaging)')
  cmd:option('-dataTime',       false,                   'if use -dataTime, use as time series data (if possible)')
  cmd:option('-preprocess',     1,                       'options: 0 | 1 (use preprocessing)')
  cmd:option('-ignoreIndex',    -100,                    'ignore specific class index in loss function')
  -- optimization options
  cmd:option('-maxEpoch',       2000,                    'max number of total epochs to run')
  cmd:option('-optimRegimePath','',                      'optimization regime path')
  cmd:option('-optimMethod',    'adam',                  'options: adam | rmsprop | sgd')
  cmd:option('-lr',             1e-3,                    'learning rate')
  cmd:option('-lrFinder',       false,                   'learning rate range finder')
  cmd:option('-lrMin',          1e-5,                    'min learning rate in range')
  cmd:option('-lrMax',          1e-2,                    'max learning rate in range')
  cmd:option('-lrScheme',       'none',                  'options: none | lin | exp | clr | cos')
  cmd:option('-lrDecayBegEpoch',20,                      'learning rate decay after #epoch')
  cmd:option('-lrDecayStep',    1,                       'learning rate decay every #epoch')
  cmd:option('-nCycle',         1,                       '#cycle in learning rate scheme')
  cmd:option('-cycleLenFactor', 1,                       'every cycle length multiply factor')
  cmd:option('-cycleLrFactor',  1,                       'every cycle learning rate range multiply factor')
  cmd:option('-weightDecay',    0,                       'weight decay for sgd and adam')
  cmd:option('-beta1',          0.9,                     'first momentum coefficient for adam')
  cmd:option('-beta2',          0.99,                    'second momentum coefficient for adam')
  cmd:option('-alpha',          0.95,                    'smoothing constant for rmsprop')
  cmd:option('-epsilon',        1e-8,                    'numerical stability for adam and rmsprop')
  cmd:option('-momentum',       0,                       'momentum for sgd')
  cmd:option('-dampening',      0,                       'dampening for momentum')
  cmd:option('-nesterov',       false,                   'nesterov for sgd')
  -- model options
  cmd:option('-network',        'FCNVGG',                'options: VGG | FCNVGG | UNet | FCSLSTMVGG')
  cmd:option('-pretrainPath',   '',                      'pretrain model path')
  cmd:option('-criterion',      'ClassNLLCriterion',     'loss function')
  cmd:option('-unbalance',      false,                   'assign weight to each of the classes in loss function')
  -- vgg model options
  cmd:option('-vggPlanes',      '64 128 256 512 512',    'vgg convolution nOutpuPlanes')
  cmd:option('-vggLayers',      '2 2 3 3 3',             'vgg convolution #layers in each block')
  cmd:option('-vggPad',         1,                       'vgg first convolution pad zero width')
  cmd:option('-vggFc',          '4096 4096',             'vgg fully-connected layers dimension')
  cmd:option('-vggNoBN',        false,                   'vgg use batch normalization or not')
  cmd:option('-vggDropout',     false,                   'vgg use dropout or not')
  cmd:option('-vggFuse',        4,                       'fcn vgg fuse level')
  cmd:option('-vggPost',        false,                   'fcn vgg use pre or post pooling fuse')
  -- unet model options
  cmd:option('-unetPlanes',     '64 128 256 512 1024',   'unet convolution nOutpuPlanes')
  cmd:option('-unetLayers',     '2 2 2 2 2',             'unet convolution #layers in each block')
  cmd:option('-unetPad',        1,                       'unet first convolution pad zero width')

  cmd:text()

  local option = cmd:parse(arg or {})
  
  local isTbl = {'gpuid', 'vggPlanes', 'vggLayers', 'vggFc', 'unetPlanes', 'unetLayers'}
  for i = 1, #isTbl do
    option[isTbl[i]] = utility.str.str2tbl(option[isTbl[i]], true)
  end
  option.manualSeed = tonumber(option.manualSeed)
  if option.datasetConfig == '' then
    option.datasetConfig = nil
  end

  torch.setdefaulttensortype('torch.' .. option.tensorType)
  
  option.action = {resume = option.resume, retrain = option.retrain}
  option.resume, option.retrain = nil, nil

  option.log = paths.concat(option.logDir, option.logStr)
  option.logStr, option.logDir = nil, nil
  if option.pastalog ~= 0 then
    option.url = 'http://localhost:' .. option.pastalog .. '/data'
    option.pastalog = nil
  end

  if option.example ~= 'none' then
    if option.exampleType ~= 'pred' and option.exampleType ~= 'allpred'
      and option.exampleType ~= 'acti' and option.exampleType ~= 'allacti'
    and option.exampleType ~= 'all' then
      error('option.exampleType options: pred|acti|allpred|allacti|all')
    end
    local set = utility.str.str2tbl(option.example, false, '%S+') -- split example string by space
    option.example = {epoch = option.exampleEpoch, type = option.exampleType, imageGap = 1}
    for i = 1, #set do
      local subset = utility.str.str2tbl(set[i], false, '[^:]+') -- split substring by colon
      if subset[1] ~= 'train' and subset[1] ~= 'test' then
        error('option.example format should be something like train:1,2 test:3,4, only train and test is supported')
      end
      local subsubset = utility.str.str2tbl(subset[2], true, '[^,]+') -- split subsubstring by comma
      if utility.tbl.len(subsubset) ~= 0 then
        option.example[subset[1]] = {}
        option.example[subset[1]].indices = subsubset
        option.example[subset[1]].n = utility.tbl.len(subsubset)
      end
    end
  else
    option.example = nil
  end
  option.exampleEpoch, option.exampleType = nil, nil

  option.gpu = {manualSeed = option.manualSeed, id = option.gpuid, ram = option.gram}
  option.manualSeed, option.gpuid, option.gram = nil, nil, nil

  option.loader = {type = option.loaderType, nThreads = option.nThreads, batchSize = option.batchSize}
  option.loaderType, option.nThreads, option.batchSize = nil, nil, nil

  option.data = {set = option.dataset, setConfig = option.datasetConfig,
                 inputType = option.dataInputType, targetType = option.dataTargetType,
                 dim = option.dataDim, time = option.dataTime, preprocess = option.preprocess,
                 ignoreIndex = option.ignoreIndex}
  option.dataset, option.datasetConfig, option.dataDim, option.dataTime = nil, nil, nil, nil
  option.dataInputType, option.dataTargetType, option.preprocess, option.ignoreIndex = nil, nil, nil, nil

  if option.lrFinder == true and option.maxEpoch > 5 then
    option.maxEpoch = 5
  end
  if option.nCycle ~= 1 then
    option.lrDecayBegEpoch = 1
    option.lrDecayStep = 1
  end

  if option.optimRegimePath == '' then
    local state
    if option.optimMethod == 'adam' then
      state = {learningRate = option.lr, weightDecay = option.weightDecay,
               beta1 = option.beta1, beta2 = option.beta2, epsilon = option.epsilon}
    elseif option.optimMethod == 'rmsprop' then
      state = {learningRate = option.lr, weightDecay = option.weightDecay,
               alpha = option.alpha, epsilon = option.epsilon}
    elseif option.optimMethod == 'sgd' then
      state = {learningRate = option.lr, weightDecay = option.weightDecay,
               momentum = option.momentum, dampening = option.dampening, nesterov = option.nesterov}
    else
      error('optim method options: adam | rmsprop | sgd')
    end
    option.optim = {method = option.optimMethod, maxEpoch = option.maxEpoch, iEpoch = 1, bestMeasure = -math.huge,
                    lrMin = option.lrMin, lrMax = option.lrMax, lrScheme = option.lrScheme,
                    lrDecayBegEpoch = option.lrDecayBegEpoch, lrDecayStep = option.lrDecayStep,
                    nCycle = option.nCycle,
                    cycleLenFactor = option.cycleLenFactor, cycleLrFactor = option.cycleLrFactor}
    option.optim.state = {[1] = state}
  else
    local regime = utility.tbl.load(option.optimRegimePath)
    local multiLR = false
    for i = 1, #regime.step do
      if regime.step[i].group ~= nil then
        multiLR = true
      end
    end
    option.optim = {method = regime.method, maxEpoch = regime.maxEpoch, iEpoch = 1, bestMeasure = -math.huge,
                    regimePath = option.optimRegimePath, regime = regime, multiLR = multiLR, state = {}}
  end
  option.maxEpoch, option.optimMethod, option.lr, option.weightDecay, option.epsilon = nil, nil, nil, nil, nil
  option.lrMin, option.lrMax, option.lrScheme, option.lrDecayBegEpoch, option.nCycle = nil, nil, nil, nil, nil
  option.lrDecayStep, option.cycleLenFactor, option.cycleLrFactor, option.optimRegimePath = nil, nil, nil, nil
  option.beta1, option.beta2, option.alpha, option.momentum, option.dampening, option.nesterov = nil, nil, nil, nil, nil, nil

  option.model = {network = option.network, pretrainPath = option.pretrainPath,
                  criterion = option.criterion, weightedLoss = option.unbalance}
  option.network, option.pretrainPath, option.criterion, option.unbalance =  nil, nil, nil, nil

  if string.find(option.model.network, 'VGG') then
    option.vgg = {planes = option.vggPlanes, layers = option.vggLayers, pad = option.vggPad,
                  fc = option.vggFc, fuse = option.vggFuse, dropout = option.vggDropout, post = option.vggPost}
    if option.vggNoBN == false then
      option.vgg.bn = true
    else
      option.vgg.bn = false
    end
  elseif string.find(option.model.network, 'UNet') then
    option.unet = {planes = option.unetPlanes, layers = option.unetLayers, pad = option.unetPad}
  else
    error('network options: VGG | FCNVGG | UNet | FCSLSTMVGG')
  end
  option.vggPlanes, option.vggLayers, option.vggPad, option.vggFc, option.vggFuse = nil, nil, nil, nil, nil
  option.vggNoBN, option.vggDropout, option.vggPost = nil, nil, nil
  option.unetPlanes, option.unetLayers, option.unetPad = nil, nil, nil

  return option
end

utility.opt = opt
