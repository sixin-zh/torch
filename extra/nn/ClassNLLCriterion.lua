local ClassNLLCriterion, parent = torch.class('nn.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function ClassNLLCriterion:updateOutput(input, target)
--   input.nn.ClassNLLCriterion_updateOutput(self, input, target)
--   return self.output

   if input:dim() == 1 then
      self.output = -input[target]
   elseif input:dim() == 2 then
      local output = torch.Tensor(1):typeAs(input)
      for i=1,target:size(1) do
         output:add(-1,input[i][{{target[i]}}])
      end
      if self.sizeAverage then
         output = output:div(target:size(1))
      end
      self.output = output[1]
   else
      error('matrix or vector expected')
   end
   return self.output

end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

--   input.nn.ClassNLLCriterion_updateGradInput(self, input, target)
--   return self.gradInput

  if input:dim() == 1 then
      self.gradInput[target] = -1
   else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      local gradInput = self.gradInput
      for i=1,target:size(1) do
         gradInput[i][target[i]] = z
      end
   end

   return self.gradInput   

end
