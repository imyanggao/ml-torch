
local CacheClsCIFAR, parent = torch.class('dataset.CacheClsCIFAR', 'dataset.Cache')

local function readBinary(binaryPath, opt)
  local binary = torch.DiskFile(binaryPath, 'r'):binary()
  -- get the number of samples
  -- check https://www.cs.toronto.edu/~kriz/cifar.html to get the binary format
  -- cifar10: 1 label byte, 3 * 32 * 32 pixel bytes
  -- cifar100: 1 coarse label byte, 1 fine label byte, 3 * 32 * 32 pixel bytes
  binary:seekEnd()
  local offset, coarseLabel, fineLabel, data
  if opt.nClass == 10 then
    offset = 1
  elseif opt.nClass == 100 then
    offset = 2
  end
  local nSample = (binary:position() - 1) / (3 * 32 * 32 + offset)
  assert(nSample == math.floor(nSample), 'number of samples should be an integer')
  binary:seek(1)

  if opt.nClass == 100 then
    coarseLabel = torch.ByteTensor(nSample)
  end
  local fineLabel = torch.ByteTensor(nSample)
  local data = torch.ByteTensor(nSample, 3, 32, 32)
  for i = 1, nSample do
    if opt.nClass == 100 then
      coarseLabel[i] = binary:readByte()
    end
    fineLabel[i] = binary:readByte()
    data[i]:copy(torch.ByteTensor(binary:readByte(3 * 32 * 32)))
  end
  
  -- original label is 0-9, which do NOT work with CrossEntropyCriterion
  if opt.nClass == 100 then
    coarseLabel:add(1)
  end
  fineLabel:add(1)
  return data, fineLabel, coarseLabel
end

local function catBinary(binaryPaths, opt)
  local data, label, coarseLabel
  for _, binaryPath in ipairs(binaryPaths) do
    local batchData, batchLabel, batchCoarseLabel = readBinary(binaryPath, opt)
    if data == nil then
      data = batchData
      label = batchLabel
      if batchCoarseLabel ~= nil then
        coarseLabel = batchCoarseLabel
      end
    else
      data = torch.cat(data, batchData, 1)
      label = torch.cat(label, batchLabel)
      if batchCoarseLabel ~= nil then
        coarseLabel = torch.cat(coarseLabel, batchCoarseLabel)
      end
    end
  end
  if opt.nClass == 10 then
    return {data = data, label = label}
  elseif opt.nClass == 100 then
    return {data = data, label = label, coarseLabel = coarseLabel}
  end
end

local function readClass(filenames)
  local classes, superclasses
  for key, filename in ipairs(filenames) do
    local file = io.open(filename)
    if key == 1 then
      classes = {}
      for line in file:lines() do
        if line ~= '' and line ~= ' ' then
          table.insert(classes, line)
        end
      end
    elseif key == 2 then
      superclasses = {}
      for line in file:lines() do
        if line ~= '' and line ~= ' ' then
          table.insert(superclasses, line)
        end
      end
    end
  end
  return classes, superclasses
end

function CacheClsCIFAR:exec()
  local trainPath, testPath, classNamePath
  if self.opt.nClass == 10 then
    trainPath = {'cifar-10-batches-bin/data_batch_1.bin',
                 'cifar-10-batches-bin/data_batch_2.bin',
                 'cifar-10-batches-bin/data_batch_3.bin',
                 'cifar-10-batches-bin/data_batch_4.bin',
                 'cifar-10-batches-bin/data_batch_5.bin'}
    testPath = {'cifar-10-batches-bin/test_batch.bin'}
    classPath = {'cifar-10-batches-bin/batches.meta.txt'}
  elseif self.opt.nClass == 100 then
    trainPath = {'cifar-100-binary/train.bin'}
    testPath = {'cifar-100-binary/test.bin'}
    classPath = {'cifar-100-binary/fine_label_names.txt',
                 'cifar-100-binary/coarse_label_names.txt'}
  else
    error('only support parameters 10 for cifar10 and 100 for cifar100')
  end
  local url = 'http://www.cs.toronto.edu/~kriz/cifar-' .. self.opt.nClass .. '-binary.tar.gz'
  print('=> Cache CIFAR-' .. self.opt.nClass .. 'dataset ... ')
  print(" | downloading dataset from " .. url)
  local ret = os.execute('curl -O ' .. url)
  assert(ret == true or ret == 0, 'error downloading CIFAR-' .. self.opt.nClass)
  print(" | unarchiving dataset")
  ret = os.execute('tar vxzf cifar-' .. self.opt.nClass .. '-binary.tar.gz')
  assert(ret == true or ret == 0, 'error unarchiving CIFAR-' .. self.opt.nClass)

  print(" | combining dataset into a single file")
  local train = catBinary(trainPath, self.opt)
  local test = catBinary(testPath, self.opt)
  local classes, superclasses = readClass(classPath)

  if superclasses ~= nil then
    self.dataset = {train = train, val = test, labels = classes, classes = classes,
                    superlabels = superclasses, superclasses = superclasses}
  else
    self.dataset = {train = train, val = test, labels = classes, classes = classes}
  end

  print(" | moving all tmp files")
  os.execute('rm -rf ' .. self.opt.path['rootDir'] .. '/cifar-' .. self.opt.nClass .. '-*')
  ret = os.execute('mv cifar-' .. self.opt.nClass .. '-* ' .. self.opt.path['rootDir'])
  assert(ret == true or ret == 0, 'error move all temp files')
end
