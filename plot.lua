require('utility.init')

local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-log',      'log-log.log',   'log file path')
cmd:option('-xCol',     '1',             'column index in log file for x axis in plot (default 1)')
cmd:option('-yCol',     '-1',            'column index in log file for y axis in plot (default all except 1st column)')
cmd:option('-logscale', false,           'logarithm scale for x axis')
cmd:text()

local option = cmd:parse(arg or {})

local nRow, nCol = 0
local value, name = {}
for line in io.lines(option.log) do
  nRow = nRow + 1
  if nRow == 1 then
    name = utility.str.str2tbl(line, false)
    nCol = #name
  else
    table.insert(value, utility.str.str2tbl(line, true))
  end
end
value = torch.Tensor(value)

local xCol = tonumber(option.xCol)
assert(xCol >= 0 and xCol <= nCol, 'xCol must be between 0 (not use any column as x axis) and #columns in log file')

local yCol
if option.yCol == '-1' then
  yCol = torch.range(2, nCol):totable()
else
  yCol = utility.str.str2tbl(option.yCol, true)
end

local lineCmd = {}
for i = 1, #yCol do
  assert(yCol[i] > 0 and yCol[i] <= nCol, 'yCol must be between 1 and #columns in log file')
  assert(yCol[i] ~= xCol, 'xCol can not be any number of yCol')
  if xCol == 0 then
    table.insert(lineCmd, {name[yCol[i]], value[{{},yCol[i]}], '-'})
  else
    table.insert(lineCmd, {name[yCol[i]], value[{{},xCol}], value[{{},yCol[i]}], '-'})
  end
end

local xlabel, ylabel
if xCol == 0 then
  xlabel = '#'
else
  xlabel = name[xCol]
end
if #yCol == 1 then
  ylabel = name[yCol[1]]
else
  ylabel = 'value'
end

local gnuplot = require('gnuplot')
if option.logscale then
  gnuplot.raw('set logscale x')
end
gnuplot.xlabel(xlabel)
gnuplot.ylabel(ylabel)
gnuplot.title(option.log)
gnuplot.plot(table.unpack(lineCmd))

