--- oTToNTToA

-- rTaoTaTaoNaTaTaoA

local argtypes = {}

argtypes.Tensor = {declare=function(arg)
                              return string.format("THTensor *arg%d = NULL;", arg.i)
                           end,

                   check=function(arg, idx)
                            return string.format("luaT_isudata(L, %d, torch_(Tensor_id))", idx)
                         end,

                   read = function(arg, idx)
                             return string.format("arg%d = luaT_toudata(L, %d, torch_(Tensor_id));", arg.i, idx)
                          end,
                   
                   carg = function(arg, idx)
                             return string.format('arg%d', arg.i)
                          end,

                   precall = function(arg)
                                local txt = {}
                                if arg.default and arg.returned then
                                   table.insert(txt, string.format('if(arg%d)', arg.i))
                                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                                   table.insert(txt, 'else')
                                   table.insert(txt, string.format('arg%d = THTensor_(new)();', arg.i))
                                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                                elseif arg.default then
                                   error('a tensor cannot be optional if not returned')
                                elseif arg.returned then
                                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                                end
                                return table.concat(txt, '\n')
                             end,

                   postcall = function(arg)
                                local txt = {}
                                if arg.creturned then
                                   -- this next line is actually debatable
                                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                                end
                                return table.concat(txt, '\n')
                             end
                }

argtypes.LongTensor = {declare=function(arg)
                              return string.format("THLongTensor *arg%d = NULL;", arg.i)
                           end,

                   check=function(arg, idx)
                            return string.format("luaT_isudata(L, %d, torch_LongTensor_id)", idx)
                         end,

                   read = function(arg, idx)
                             return string.format("arg%d = luaT_toudata(L, %d, torch_LongTensor_id);", arg.i, idx)
                          end,
                   
                   carg = function(arg, idx)
                             return string.format('arg%d', arg.i)
                          end,

                   precall = function(arg)
                                local txt = {}
                                if arg.default and arg.returned then
                                   table.insert(txt, string.format('if(arg%d)', arg.i))
                                   table.insert(txt, string.format('THLongTensor_retain(arg%d);', arg.i))
                                   table.insert(txt, 'else')
                                   table.insert(txt, string.format('arg%d = THLongTensor_new();', arg.i))
                                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_LongTensor_id);', arg.i))
                                elseif arg.default then
                                   error('a tensor cannot be optional if not returned')
                                elseif arg.returned then
                                   table.insert(txt, string.format('THLongTensor_retain(arg%d);', arg.i))
                                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_LongTensor_id);', arg.i))
                                end
                                return table.concat(txt, '\n')
                             end,

                   postcall = function(arg)
                                 local txt = {}
                                 if arg.creturned then
                                    -- this next line is actually debatable
                                    table.insert(txt, string.format('THLongTensor_retain(arg%d);', arg.i))
                                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_LongTensor_id);', arg.i))
                                 end
                                 return table.concat(txt, '\n')
                              end
}

argtypes.integer = {declare=function(arg)
                               return string.format("long arg%d = %d;", arg.i, arg.default or 0)
                            end,

                    check = function(arg, idx)
                               return string.format("lua_isnumber(L, %d)", idx)
                            end,

                    read = function(arg, idx)
                              return string.format("arg%d = (long)lua_tonumber(L, %d)-1;", arg.i, idx)
                           end,

                    carg = function(arg, idx)
                              return string.format('arg%d', arg.i)
                           end,

                    precall = function(arg)
                                 if arg.returned then
                                    return string.format('lua_pushnumber(L, (lua_Number)arg%d+1);', arg.i)
                                 end
                              end,

                    postcall = function(arg)
                                  if arg.creturned then
                                     return string.format('lua_pushnumber(L, (lua_Number)arg%d+1);', arg.i)
                                  end
                               end
                 }

argtypes.real = {declare=function(arg)
                               return string.format("real arg%d = %d;", arg.i, arg.default or 0)
                            end,

                    check = function(arg, idx)
                               return string.format("lua_isnumber(L, %d)", idx)
                            end,

                    read = function(arg, idx)
                              return string.format("arg%d = (real)lua_tonumber(L, %d);", arg.i, idx)
                           end,

                    carg = function(arg, idx)
                              return string.format('arg%d', arg.i)
                           end,

                    precall = function(arg)
                                 if arg.returned then
                                    return string.format('lua_pushnumber(L, (lua_Number)arg%d);', arg.i)
                                 end
                              end,

                    postcall = function(arg)
                                  if arg.creturned then
                                     return string.format('lua_pushnumber(L, (lua_Number)arg%d);', arg.i)
                                  end
                               end
                 }

argtypes.boolean = {declare=function(arg)
                               local default = 0
                               if arg.default then
                                  default = 1
                               end
                               return string.format("int arg%d = %d;", arg.i, default or 0)
                            end,

                    check = function(arg, idx)
                               return string.format("lua_isboolean(L, %d)", idx)
                            end,

                    read = function(arg, idx)
                              return string.format("arg%d = lua_boolean(L, %d);", arg.i, idx)
                           end,

                    carg = function(arg, idx)
                              return string.format('arg%d', arg.i)
                           end,

                    precall = function(arg)
                                 if arg.returned then
                                    return string.format('lua_pushboolean(L, arg%d);', arg.i)
                                 end
                              end,

                    postcall = function(arg)
                                  if arg.creturned then
                                     return string.format('lua_pushboolean(L, arg%d);', arg.i)
                                  end
                               end
                 }

local function bit(p)
   return 2 ^ (p - 1)  -- 1-based indexing                                                          
end

local function hasbit(x, p)
   return x % (p + p) >= p
end

local function beautify(txt)
   local indent = 0
   for i=1,#txt do
      if txt[i]:match('}') then
         indent = indent - 2
      end
      if indent > 0 then
         txt[i] = string.rep(' ', indent) .. txt[i]
      end
      if txt[i]:match('{') then
         indent = indent + 2
      end
   end
end

function generateinterface(luafuncname, cfuncname, args)
   local txt = {}

   table.insert(txt, string.format("static int %s(lua_State *L)", luafuncname))
   table.insert(txt, "{")
   table.insert(txt, "int narg = lua_gettop(L);")

   local nopt = 0
   local txtargs = {}
   local cargs = {}
   local argcreturned
   for i,arg in ipairs(args) do
      arg.i = i
      assert(argtypes[arg.name], 'unknown type ' .. arg.name)
      table.insert(txt, argtypes[arg.name].declare(arg))
      local name = arg.name
      if arg.returned then
         name = string.format('*%s*', name)
      end
      if arg.default then
         nopt = nopt + 1
         table.insert(txtargs, string.format('[%s]', name))
      elseif not arg.creturned then
         table.insert(txtargs, name)
      end
      if arg.creturned then
         if argcreturned then
            error('A C function can only return one argument!')
         end
         if arg.default then
            error('Obviously, an "argument" returned by a C function cannot have a default value')
         end
         if arg.returned then
            error('Options "returned" and "creturned" are incompatible')
         end
         argcreturned = arg
      else
         table.insert(cargs, argtypes[arg.name].carg(arg))
      end
   end

   for variant=0,math.pow(2, nopt)-1 do
      local opt = 0
      local currentargs = {}
      for i,arg in ipairs(args) do
         if arg.default then
            opt = opt + 1
            if hasbit(variant, bit(opt)) then
               table.insert(currentargs, arg)
            end
         elseif not arg.creturned then
            table.insert(currentargs, arg)
         end
      end

      if variant == 0 then
         table.insert(txt, string.format('if(narg == %d', #currentargs))
      else
         table.insert(txt, string.format('else if(narg == %d', #currentargs))
      end

      for stackidx, arg in ipairs(currentargs) do
         table.insert(txt, string.format("&& %s", argtypes[arg.name].check(arg, stackidx)))
      end
      table.insert(txt, ')')
      table.insert(txt, '{')

      for stackidx, arg in ipairs(currentargs) do
         table.insert(txt, argtypes[arg.name].read(arg, stackidx))
      end

      table.insert(txt, '}')

   end

   table.insert(txt, 'else')
   table.insert(txt, string.format('luaL_error(L, "expected arguments: %s");', table.concat(txtargs, ' ')))

   for _,arg in ipairs(args) do
      local precall = argtypes[arg.name].precall(arg)
      if not precall or not precall:match('^%s*$') then
         table.insert(txt, precall)
      end
   end

   if argcreturned then
      table.insert(txt, string.format('arg%d = %s(%s);', argcreturned.i, cfuncname, table.concat(cargs, ',')))
   else
      table.insert(txt, string.format('%s(%s);', cfuncname, table.concat(cargs, ',')))
   end

   for _,arg in ipairs(args) do
      local postcall = argtypes[arg.name].postcall(arg)
      if not postcall or not postcall:match('^%s*$') then
         table.insert(txt, postcall)
      end
   end

   local nret = 0
   if argcreturned then
      nret = nret + 1
   end
   for _,arg in ipairs(args) do
      if arg.returned then
         nret = nret + 1
      end
   end
   table.insert(txt, string.format('return %d;', nret))
   table.insert(txt, '}')
   
--   beautify(txt)
   print(table.concat(txt, '\n'))
   
end

-- generateinterface("zero", {{name="tensor", returned=true}})

-- generateinterface("cross", {{name="tensor", default=true, returned=true},
--                             {name="tensor"},
--                             {name="tensor"},
--                             {name="integer", default=0}})

-- generateinterface("cadd", {{name="tensor", default=true, returned=true},
--                            {name="tensor"},
--                            {name="real", default=1},
--                            {name="tensor"}})

-- generateinterface("fill", {{name="tensor", returned=true},
--                            {name="real"}})

-- generateinterface("dot", {{name="tensor", returned=true},
--                            {name="real"}})

-- generateinterface("addcmul", {{name="tensor", default=true, returned=true},
--                               {name="tensor"},
--                               {name="real", default=1},
--                               {name="tensor"},
--                               {name="tensor"}})

-- generateinterface("addmv", {{name="tensor", default=true, returned=true},
--                             {name="real", default=1},
--                             {name="tensor"},
--                             {name="real", default=1},
--                             {name="tensor"},
--                             {name="tensor"}})

-- generateinterface("min", {{name="Tensor", default=true, returned=true},
--                           {name="LongTensor", default=true, returned=true},
--                           {name="Tensor"},
--                           {name="integer", default=0}})

local function lname(name)
   return string.format('lab_(%s)', name)
end

local function cname(name)
   return string.format('THLab_(%s)', name)
end

-- generateinterface(luaname("sort"),
--                   cname("sort"),
--                   {{name="Tensor", default=true, returned=true},
--                    {name="LongTensor", default=true, returned=true},
--                    {name="Tensor"},
--                    {name="integer", default=0},
--                    {name="boolean", default=0}})

generateinterface(lname("dot"),
                  cname("dot"),
                  {{name="Tensor"},
                   {name="Tensor"},
                   {name="real", creturned=true}})

--      luaL_error(L, "invalid arguments: [tensor longtensor] tensor [dimension]"); \