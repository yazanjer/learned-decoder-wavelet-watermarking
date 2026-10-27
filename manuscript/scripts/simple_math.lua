-- V1.0 Word build: render simple inline TeX math (numbers, signs, intervals, exponents, Greek letters,
-- single-letter symbols) as plain Unicode text; complex math stays as Word equations.
local sup = {["0"]="⁰",["1"]="¹",["2"]="²",["3"]="³",["4"]="⁴",["5"]="⁵",["6"]="⁶",["7"]="⁷",["8"]="⁸",["9"]="⁹",["-"]="⁻",["+"]="⁺"}
local rep = {{"\\pm","±"},{"\\times","×"},{"\\sigma","σ"},{"\\theta","θ"},{"\\hat\\theta","θ̂"},{"\\alpha","α"},
  {"\\kappa","κ"},{"\\epsilon","ε"},{"\\zeta","ζ"},{"\\rho","ρ"},{"\\mathrm{Holm}","Holm"},{"\\le","≤"},{"\\ge","≥"},
  {"\\approx","≈"},{"\\circ","°"},{"\\,"," "},{"\\!",""},{"\\ ",""},{"{,}",","},{"\\%","%"}}
function Math(el)
  if el.mathtype ~= "InlineMath" then return nil end
  local t = el.text
  if t:find("\\frac") or t:find("\\sum") or t:find("\\mathbb") or t:find("\\mathcal") or t:find("\\lVert") or t:find("\\left") then return nil end
  for _, r in ipairs(rep) do t = t:gsub(r[1]:gsub("%p","%%%0"), r[2]) end
  t = t:gsub("%^{([%-%+%d]+)}", function(s) return (s:gsub(".", sup)) end)
  t = t:gsub("%^([%d])", function(s) return sup[s] end)
  t = t:gsub("_{([^}]*)}", "%1"):gsub("_(%w)", "%1")
  if t:find("[\\{}%^]") then return nil end
  t = t:gsub("^%-", "−"):gsub("([%s%[,(])%-", "%1−"):gsub("^%-$", "−")
  if t == "" then t = "−" end
  if t:match("^%a$") then return pandoc.Emph{pandoc.Str(t)} end
  return pandoc.Str(t)
end
