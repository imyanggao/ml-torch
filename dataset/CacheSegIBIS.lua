
local ffi = require 'ffi'

local CacheSegIBIS, parent = torch.class('dataset.CacheSegIBIS', 'dataset.Cache')

local function labelMap()
  local classes = {[0] = 'bg', [85] = 'csf', [170] = 'gray', [255] = 'white'}
  local label2class = {[1] = 0, [2] = 85, [3] = 170, [4] = 255}
  local class2label= {[0] = 1, [85] = 2, [170] = 3, [255] = 4}
  local labels = {}
  for class, name in pairs(classes) do
    labels[class2label[class]] = name
  end
  return labels, classes, label2class, class2label
end

local function rename(opt)
  local cachePath = utility.io.splitPath(opt.path['cacheFile'])
  local renameRoot = paths.concat(cachePath, 'rename')
  utility.sys.mkdir(renameRoot)
  local extension = opt.ext
  local originNames = utility.io.findByExts(opt.path['originDir'], extension)
  local idxSub, idxMon, subject, month, nameMap = {}, {}, {}, {}, {}
  local name, subjectStr, monthStr, typeStr, newName
  for i = 1, #originNames do
    local path, basename, _ = utility.io.splitPath(originNames[i])
    name = paths.concat(path, basename) .. '.'
    subjectStr = string.format('%06d', name:match('(%d+)_V'))
    monthStr = string.format('%02d', name:match('_V(%d%d)'))
    if idxSub[subjectStr] == nil then
      idxSub[subjectStr] = utility.tbl.len(idxSub) + 1
      subject[idxSub[subjectStr]] = subjectStr
      idxMon[subjectStr] = {}
      month[idxSub[subjectStr]] = {}
    end
    if idxMon[subjectStr][monthStr] == nil then
      idxMon[subjectStr][monthStr] = 1
      table.insert(month[idxSub[subjectStr]], monthStr)
    end
    if string.find(name, 'labels') then typeStr = 'LA'
    elseif string.find(name, 't1') then typeStr = 'T1'
    elseif string.find(name, 't2') then typeStr = 'T2'
    else error("could not figure out type") end
    newName = renameRoot .. '/S' .. subjectStr .. '-M' .. monthStr .. '-' .. typeStr .. '.'
    utility.sys.exec('c3d ' .. name .. extension .. ' -o ' .. newName .. 'nii')
    nameMap[newName .. 'nii'] = name .. extension
    print(' | S' .. subjectStr .. '-M' .. monthStr .. '-' .. typeStr)
  end
  for i = 1, #month do
    table.sort(month[i])
  end
  return subject, month, nameMap
end

-- cached dataset: nSubject * nTime * nModality (T1, T2, LA) * slice * height * width
local function read(opt, subject, month, nameMap)
  local cachePath = utility.io.splitPath(opt.path['cacheFile'])
  local renameRoot = paths.concat(cachePath, 'rename')
  local loader3d, nameMapNew, cached, old, new = utility.img3d.load(cachePath), {}
  for i = 1, #month do
    for j = 1, #month[i] do
      local oldName = renameRoot .. '/S' .. subject[i] .. '-M' .. month[i][j] .. '-'
      local newName = renameRoot .. '/' .. string.format('%02d', i) .. '-t' .. string.format('%d', j) .. '-'
      for k, v in ipairs({'T1', 'T2', 'LA'}) do
        old = oldName .. v .. '.nii'
        new = newName .. v .. '.nii'
        utility.sys.exec('mv ' .. old .. ' ' .. new)
        nameMapNew[string.format('%02d', i) .. '-t' .. string.format('%d', j) .. '-' .. v] = nameMap[old]
        local img3d = loader3d(new)
        if i == 1 and j == 1 and v == 'T1' then
          cached = torch.ByteTensor(#subject, #month[1], 3, table.unpack(img3d:size():totable()))
        end
        cached[i][j][k] = img3d
        print(' | ' .. string.format('%02d', i) .. '-t' .. string.format('%d', j) .. '-' .. v)
      end
    end
    collectgarbage()
  end
  return cached, nameMapNew
end

-- apply label mapping to transfer label from 1 to #class
local function applyLabelMap(cached, labelMapping)
  local labelTensor = cached[{{},{},{3},{},{},{}}]
  local mapping = utility.img.transLabels(labelMapping)
  cached[{{},{},{3},{},{},{}}] = mapping(labelTensor)
  collectgarbage()
  return cached
end

function CacheSegIBIS:exec()
  if self.opt.path['originDir'] == nil or self.opt.ext == nil then
    error('need to specify opt: path[originDir] path[cacheFile] and ext')
  else
    print('=> Cache IBIS dataset ... ')
    print(self.opt)
  end
  print(' | load label mapping')
  local labels, classes, label2class, class2label = labelMap()
  print(' | rename 3D images')
  local subject, month, tmpMap = rename(self.opt)
  print(' | cache 3D images')
  local cached, nameMap = read(self.opt, subject, month, tmpMap)
  print(' | apply label mapping')
  cached = applyLabelMap(cached, class2label)
  self.dataset = {cache = cached, labels = labels, classes = classes,
                  label2class = label2class, class2label = class2label,
                  subject = subject, month = month, nameMap = nameMap}
end
