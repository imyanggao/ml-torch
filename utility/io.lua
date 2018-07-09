require 'lfs'

local io = {}

function io.type(path)
  if path ~= nil then
    local type = lfs.attributes(path, "mode")
    return type
  else
    return nil
  end
end

-- check path and its all subdirectories
-- input exts could be a string or a table of strings
function io.findByExts(path, exts, names)
  if names == nil then
    names = {}
  end
  if type(exts) == 'string' then
    exts = {exts}
  end
  local sorted = {}
  for entity in lfs.dir(path) do
    sorted[#sorted+1] = entity
  end
  table.sort(sorted)
  local slash_fill = ""
  if string.sub(path, path:len(), path:len()) ~= "/" then
    slash_fill = "/"
  end
  for k, entity in pairs(sorted) do
    if entity ~= "." and entity ~= ".." then
      local fullpath = path .. slash_fill .. entity
      local type = io.type(fullpath)
      if type == "file" then
        for _, ext in pairs(exts) do
          if string.find(entity, ext) then
            names[#names + 1] = fullpath
          end
        end
      elseif type == "directory" then
        io.findByExts(fullpath, exts, names)
      end
    end
  end
  return names
end

function io.splitPath(path)
  local dir, base, ext
  if string.find(path, '/') == nil then
    dir = '.'
    if string.find(path, '%.') == nil then
      base = path
      ext = nil
    else
      base = string.gsub(path, '(.*)%.(.*)', '%1')
      ext = string.gsub(path, '(.*)%.(.*)', '%2')
    end
  else
    if string.find(path, '(.*)/(.*)%.(.*)') == nil then
      dir = string.gsub(path, '(.*)/(.*)', '%1')
      base = string.gsub(path, '(.*)/(.*)', '%2')
      ext = nil
    else
      dir = string.gsub(path, '(.*)/(.*)%.(.*)', '%1')
      base = string.gsub(path, '(.*)/(.*)%.(.*)', '%2')
      ext = string.gsub(path, '(.*)/(.*)%.(.*)', '%3')
    end
  end
  return dir, base, ext
end

utility.io = io
