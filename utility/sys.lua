
local sys = {}

function sys.exec(cmd)
  cmd = cmd .. ' 2>&1'
  local f = io.popen(cmd)
  local s = f:read('*all')
  assert(f:close(), 'Please check command: ' .. cmd)
  s = s:gsub('^%s*',''):gsub('%s*$','')
  return s
end

function sys.mkdir(dir)
  sys.exec('mkdir -p ' .. dir)
end

function sys.rm(path)
  sys.exec('rm -rf ' .. path)
end

function sys.date()
  return os.date("%Y%m%d_%H%M%S")
end

utility.sys = sys
