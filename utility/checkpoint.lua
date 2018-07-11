
local checkpoint = {}

function checkpoint.save(model, option)
  -- model = utility.net.copyNewMetatable(model):float():clearState() -- a clean copy on CPU, BUG: luajit not enough memory
  local clearModel              -- a clean copy on CPU
  if torch.type(model) == 'nn.DataParallelTable' then
    clearModel = model:get(1):clone():type(option.tensorType)
  else
    clearModel = model:clone():type(option.tensorType)
  end
  local latestModelPath, bestModelPath = option.log .. '-save-latest-model.t7', option.log .. '-save-best-model.t7'
  local latestOptimPath, bestOptimPath = option.log .. '-save-latest-optim.t7', option.log .. '-save-best-optim.t7'
  local timer = torch.Timer()
  if option.optim.bestEpoch == option.optim.iEpoch then
    print('=> Saving best checkpoint on latest')
    option.optim.bestModel = latestModelPath
    if paths.filep(bestModelPath) then
      utility.sys.exec('rm ' .. bestModelPath)
      utility.sys.exec('rm ' .. bestOptimPath)
    end
  else
    option.optim.bestModel = bestModelPath
    print('=> Saving best checkpoint by moving')
    timer:reset()
    utility.sys.exec('mv ' .. latestModelPath .. ' ' .. bestModelPath)
    utility.sys.exec('mv ' .. latestOptimPath .. ' ' .. bestOptimPath)
    print('takes ' .. timer:time().real .. ' seconds')
  end
  print('=> Saving latest checkpoint')
  timer:reset()
  torch.save(latestModelPath, clearModel)
  torch.save(latestOptimPath, option.optim)
  print('takes ' .. timer:time().real .. ' seconds')
  clearModel = nil
  collectgarbage()
end

function checkpoint.load(option)
  local timer, start = torch.Timer()
  if option.action.resume == true then
    local latestModelPath, latestOptimPath = option.log .. '-save-latest-model.t7', option.log .. '-save-latest-optim.t7'
    if not paths.filep(latestModelPath) then
      error(latestModelPath .. ' does not exist')
    else
      print('=> Resuming from latest checkpoint: ' .. latestModelPath)
      print('=> Loading latest checkpoint')
      timer:reset()
      local model = torch.load(latestModelPath)
      local optim = torch.load(latestOptimPath)
      print('takes ' .. timer:time().real .. ' seconds')
      optim.maxEpoch = option.optim.maxEpoch
      option.optim = optim
      option.optim.iEpoch = option.optim.iEpoch + 1
      return model
    end
  elseif option.action.retrain ~= '' then
    if not paths.filep(option.action.retrain) then
      error(option.action.retrain .. ' does not exist')
    else
      print('=> Retrain from the checkpoint: ' .. option.action.retrain)
      print('=> Loading checkpoint: ' .. option.action.retrain)
      timer:reset()
      local model = torch.load(option.action.retrain)
      print('takes ' .. timer:time().real .. ' seconds')
      return model
    end
  else
    return nil
  end
end

utility.checkpoint = checkpoint
