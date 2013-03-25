local SpatialConvolutionBatch, parent = torch.class('nn.SpatialConvolutionBatch', 'nn.Module')

function SpatialConvolutionBatch:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane) -- TODO
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function SpatialConvolutionBatch:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end) 
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function SpatialConvolutionBatch:updateOutput(input)
   return input.nn.SpatialConvolutionBatch_updateOutput(self, input)
end

function SpatialConvolutionBatch:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionBatch_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionBatch:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionBatch_accGradParameters(self, input, gradOutput, scale)
end

-- this routine copies weight+bias from a regular SpatialConvolution module
function SpatialConvolutionBatch:copy(sc)
   self.weight:copy(sc.weight)
   self.bias:copy(sc.bias)
end
