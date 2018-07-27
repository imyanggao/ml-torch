
local str = {}

-- pad strg to len with padchar from pos: 'l' (left), 'r' (right)
-- if strg is longer than len, no pad, return origin strg, false
-- otherwise return the padstrg, true
function str.pad(strg, len, padchar, pos)
  assert(pos == 'l' or pos == 'r', 'pos could only be l(left) or r(right)') 
  local pad, padstrg = string.rep(padchar, len - #strg)
  if pos == 'l' then
    padstrg = pad .. strg
  elseif pos == 'r' then
    padstrg = strg .. pad
  end
  return padstrg, padstrg ~= strg
end

function str.lpad(strg, len, padchar)
  padchar = padchar or ' '
  local padstrg, success = str.pad(strg, len, padchar, 'l')
  return padstrg, success
end

function str.rpad(strg, len, padchar)
  padchar = padchar or ' '
  local padstrg, success = str.pad(strg, len, padchar, 'r')
  return padstrg, success
end

function str.cpad(strg, len, padchar)
  padchar = padchar or ' '
  local padstrg, success1 = str.lpad(strg, (len + #strg) / 2, padchar)
  local padstrg, success2 = str.rpad(padstrg, len, padchar)
  return padstrg, success1 or success2
end

function str.fixLen(strg, len)
  local fixLenStrg
  if #strg > len then
    fixLenStrg = string.sub(strg, 1, len)
  else
    fixLenStrg = str.rpad(strg, len)
  end
  return fixLenStrg
end

function str.str2tbl(strg, isnum, delimiter)
  if isnum == nil then
    isnum = false
  end
  delimiter = delimiter or '%S+'
  local tbl = {}
  for i in string.gmatch(strg, delimiter) do
    if isnum == true then
      i = tonumber(i)
    end
    table.insert(tbl, i)
  end
  return tbl
end

utility.str = str
