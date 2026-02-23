module github.com/bzoidberg/heuristic-secrets/experiments/yamit/tokenizer

go 1.23.0

require (
	github.com/sugarme/tokenizer v0.3.0
	github.com/valyala/fastjson v1.6.7
	golang.org/x/text v0.25.0
)

replace github.com/sugarme/tokenizer => github.com/aeryncaen/tokenizer v0.0.0-20260223110315-2f03327a22c4

replace github.com/dlclark/regexp2 => github.com/aeryncaen/regexp2 v0.0.0-20260223063553-3a989801f37f

require (
	github.com/dlclark/regexp2 v1.11.5 // indirect
	github.com/emirpasic/gods v1.18.1 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/patrickmn/go-cache v2.1.0+incompatible // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/schollz/progressbar/v2 v2.15.0 // indirect
	golang.org/x/sync v0.14.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
