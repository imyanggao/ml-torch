
local Cache = torch.class('dataset.Cache')

function Cache:__init(opt)
  self.opt = opt
  self.dataset = {}
end

function Cache:exec()
  
end

function Cache:save()
  local path = utility.io.splitPath(self.opt.path['cacheFile'])
  print(" | create a new cache directory: " .. path)
  utility.sys.rm(path)
  utility.sys.mkdir(path)
  self:exec()
  print(" | saving dataset into " .. self.opt.path['cacheFile'])
  torch.save(self.opt.path['cacheFile'], self.dataset)
end
