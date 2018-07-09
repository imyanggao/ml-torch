
local ffi = require 'ffi'

local CacheSegBRIC, parent = torch.class('dataset.CacheSegBRIC', 'dataset.Cache')

local function labelMap()
  local classes = {[0] = 'bg', [10] = 'csf', [153] = 'gray', [255] = 'white'}
  local label2class = {[1] = 0, [2] = 10, [3] = 153, [4] = 255}
  local class2label= {[0] = 1, [10] = 2, [153] = 3, [255] = 4}
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
  local ext1, ext2 = opt.ext[1], opt.ext[2]
  local originNames = utility.io.findByExts(opt.path['originDir'], ext1)
  local idxSub, idxMon, subject, month, nameMap = {}, {}, {}, {}, {}
  local name, subjectStr, monthStr, typeStr, newName
  for i = 1, #originNames do
    local path, basename, _ = utility.io.splitPath(originNames[i])
    name = paths.concat(path, basename) .. '.'
    subjectStr = string.format('%02d', name:match('/M(%d+)-'))
    monthStr = string.format('%02d', name:match('_A(%d+)month'))
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
    if string.find(name, '-T1') then typeStr = 'T1'
    elseif string.find(name, '-T2') then typeStr = 'T2'
    elseif string.find(name, '-manual') then typeStr = 'LM'
    elseif string.find(name, '-seg') then typeStr = 'LA'
    else error("could not figure out type") end
    newName = renameRoot .. '/S' .. subjectStr .. '-M' .. monthStr .. '-' .. typeStr .. '.'
    utility.sys.exec('cp ' .. name .. ext1 .. ' ' .. newName .. ext1)
    utility.sys.exec('cp ' .. name .. ext2 .. ' ' .. newName .. ext2)
    nameMap[newName .. ext1] = name .. ext1
  end
  for i = 1, #month do
    table.sort(month[i])
  end
  return subject, month, nameMap
end

-- cached dataset: nSubject * nTime * nModality (T1, T2, LA, LM) * slice * height * width
local function read(opt, subject, month, nameMap)
  local cachePath = utility.io.splitPath(opt.path['cacheFile'])
  local renameRoot = paths.concat(cachePath, 'rename')
  local loader3d, nameMapNew, sliceLM, cached, old, new = utility.img3d.load(cachePath), {}, {}
  local ext1, ext2 = opt.ext[1], opt.ext[2]
  for i = 1, #month do
    sliceLM[i] = {}
    for j = 1, #month[i] do
      local oldName = renameRoot .. '/S' .. subject[i] .. '-M' .. month[i][j] .. '-'
      local newName = renameRoot .. '/' .. string.format('%02d', i) .. '-t' .. string.format('%d', j) .. '-'
      for k, v in ipairs({'T1', 'T2', 'LA', 'LM'}) do
        old = oldName .. v .. '.'
        new = newName .. v .. '.'
        utility.sys.exec('mv ' .. old .. ext1 .. ' ' .. new .. ext1)
        utility.sys.exec('mv ' .. old .. ext2 .. ' ' .. new .. ext2)
        nameMapNew[string.format('%02d', i) .. '-t' .. string.format('%d', j) .. '-' .. v] = nameMap[old .. ext1]
        local img3d = loader3d(new .. ext1)
        if i == 1 and j == 1 and v == 'T1' then
          cached = torch.ByteTensor(#subject, #month[1], 4, table.unpack(img3d:size():totable()))
        end
        cached[i][j][k] = img3d
        if v == 'LM' then
          sliceLM[i][j] = utility.img3d.boundary(img3d)[1][1]
        end
          print(' | ' .. string.format('%02d', i) .. '-t' .. string.format('%d', j) .. '-' .. v)
      end
    end
  end
  return cached, nameMapNew, sliceLM
end

-- bugs in manual labeled with outlier intensity = 112, correct intensity = 153
local function fix(cached, labelMapping)
  local labelTensor = cached[{{},{},{3,4},{},{},{}}]
  labelTensor[labelTensor:eq(112)] = 153
  local mapping = utility.img.transLabels(labelMapping)
  cached[{{},{},{3,4},{},{},{}}] = mapping(labelTensor)
  collectgarbage()
  return cached
end

function CacheSegBRIC:exec()
  if self.opt.path['originDir'] == nil or self.opt.ext == nil then
    error('need to specify opt: path[originDir] path[cacheFile], ext[1] and ext[2]')
  else
    print('=> Cache BRIC dataset ... ')
    print(self.opt)
  end
  print(' | load label mapping')
  local labels, classes, label2class, class2label = labelMap()
  print(' | rename 3D images')
  local subject, month, tmpMap = rename(self.opt)
  print(' | cache 3D images')
  local cached, nameMap, sliceLM = read(self.opt, subject, month, tmpMap)
  print(' | fix manual label bug and label mapping')
  cached = fix(cached, class2label)
  self.dataset = {cache = cached, labels = labels, classes = classes, sliceLM = sliceLM,
                  label2class = label2class, class2label = class2label,
                  subject = subject, month = month, nameMap = nameMap}
end
