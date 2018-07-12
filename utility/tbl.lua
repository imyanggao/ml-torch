
local tbl = {}

function tbl.len(tbl)
  local count = 0
  for _ in pairs(tbl) do count = count + 1 end
  return count
end

-- must be index-value arrarys for the input parameters
function tbl.cat(...)
  local tbls = {...}
  local idx, all = 1, {}
  for i, v in ipairs(tbls) do
    if type(v) == 'table' then
      for i, v in ipairs(v) do
        table.insert(all, v)
        idx = idx + 1
      end
    else
      table.insert(all, v)
      idx = idx + 1
    end
  end
  return all
end

function tbl.inv(tbl)
  local invTbl = {}
  for k, v in pairs(tbl) do
    invTbl[v] = k
  end
  return invTbl
end

-- may cause stack overflow
function tbl.deepClone(input)
  local inputType = type(input)
  local copy
  if inputType == 'table' then
    copy = {}
    for k, v in next, input, nil do
      copy[tbl.deepClone(k)] = tbl.deepClone(v)
    end
    setmetatable(copy, tbl.deepClone(getmetatable(input)))
  else
    copy = input                -- for number, string, boolean, etc
  end
  return copy
end

function tbl.shallowCopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function tbl.tsrClone(tbl, resetZero)
  local clone = {}
  for k,v in pairs(tbl) do
    clone[k] = v:clone()
    if resetZero then
      clone[k]:zero()
    end
  end
  return clone
end

-- base on the code from http://lua-users.org/wiki/SaveTableToFile
-- tbl.save and tbl.load should be used as pairs
function tbl.save(filename, input)
  -- returns a "Lua" portable version of the string
  -- option %q puts quotes around a string argument's value
  local function exportstring(s)
    return string.format("%q", s)
  end

  local charS, charE = "   ", "\n"
  local file, err = io.open(filename, "wb")
  if err then return err end
  
  -- initiate variables for save procedure
  local tables, lookup = {input}, {[input] = 1}
  file:write("return {" .. charE)

  for idx, t in ipairs(tables) do
    file:write("-- Table: {" .. idx .. "}" .. charE)
    file:write("{" .. charE)
    local thandled = {}
    
    for i, v in ipairs(t) do
      thandled[i] = true
      local stype = type(v)
      -- only handle value
      if stype == "table" then
        if not lookup[v] then
          table.insert(tables, v)
          lookup[v] = #tables
        end
        file:write(charS .. "{" .. lookup[v] .. "}," .. charE)
      elseif stype == "string" then
        file:write(charS .. exportstring(v) .. "," .. charE)
      elseif stype == "number" then
        file:write(charS .. tostring(v) .. "," .. charE)
      end
    end
    
    for i, v in pairs(t) do
      -- escape handled values
      if (not thandled[i]) then
        local str = ""
        local stype = type(i)
        -- handle index
        if stype == "table" then
          if not lookup[i] then
            table.insert(tables,i)
            lookup[i] = #tables
          end
          str = charS .. "[{" .. lookup[i] .. "}]="
        elseif stype == "string" then
          str = charS .. "[" .. exportstring(i) .. "]="
        elseif stype == "number" then
          str = charS .. "[" .. tostring(i) .. "]="
        end
        
        if str ~= "" then
          stype = type(v)
          -- handle value
          if stype == "table" then
            if not lookup[v] then
              table.insert(tables, v)
              lookup[v] = #tables
            end
            file:write(str .. "{" .. lookup[v] .. "}," .. charE)
          elseif stype == "string" then
            file:write(str .. exportstring(v) .. "," .. charE)
          elseif stype == "number" then
            file:write(str .. tostring(v) .. "," .. charE)
          end
        end
      end
    end
    file:write("}," .. charE)
  end
  file:write("}")
  file:close()
end

function tbl.load(filename)
  local ftables, err = loadfile(filename)
  if err then return _, err end
  local tables = ftables()
  for idx = 1, #tables do
    local tolinki = {}
    for i, v in pairs(tables[idx]) do
      if type(v) == "table" then
        tables[idx][i] = tables[v[1]]
      end
      if type(i) == "table" and tables[i[1]] then
        table.insert(tolinki, {i, tables[i[1]]})
      end
    end
    -- link indices
    for _, v in ipairs(tolinki) do
      tables[idx][v[2]], tables[idx][v[1]] =  tables[idx][v[1]], nil
    end
  end
  return tables[1]
end

utility.tbl = tbl
