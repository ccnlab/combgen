// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/kit"
)

// PatsType is the type of training patterns
type PatsType int32

//go:generate stringer -type=PatsType

var KiT_PatsType = kit.Enums.AddEnum(PatsTypeN, kit.NotBitFlag, nil)

func (ev PatsType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *PatsType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// Horizontal and Vertical Lines to localist ID units
	LinesToID PatsType = iota

	// Horizontal and Vertical Lines to lines in the output
	LinesToLines

	// Random input / output patterns
	Random

	PatsTypeN
)

// CombEnv is a combinatorial environment where patterns are composed from
// independent elements.  If the network learns the separate elements, it
// can rapidly generalize to the entire set.
type CombEnv struct {
	Nm         string          `desc:"name of this environment"`
	Dsc        string          `desc:"description of this environment"`
	PatsType   PatsType        `desc:"type of patterns to use"`
	Test       bool            `desc:"present testing patterns"`
	NPools     int             `desc:"number of independent pools of patterns"`
	NTrain     int             `desc:"number of training items to use"`
	NTest      int             `desc:"number of testing items to use"`
	PatsSize   evec.Vec2i      `desc:"size of each pattern"`
	NPats      int             `desc:"number of patterns -- must be set for random patterns"`
	RndPctOn   float32         `desc:"proportion activity for random patterns"`
	RndMinDiff float32         `desc:"proportion minimum difference for random patterns"`
	Pats       etable.Table    `view:"no-inline" desc:"patterns for each pool"`
	Order      []int           `desc:"order of items to present"`
	TrainItems etensor.Int     `view:"no-inline" desc:"training patterns [NTrain][NPools]"`
	TestItems  etensor.Int     `view:"no-inline" desc:"testing patterns [NTest][NPools]"`
	Input      etensor.Float32 `view:"no-inline" desc:"current input state, 4D 1 x Size x PatSize"`
	Output     etensor.Float32 `view:"no-inline" desc:"current output state, 4D 1 x Size x Output Size"`
	Run        env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial      env.Ctr         `view:"inline" desc:"trial increments over input states -- could add Event as a lower level"`
}

func (ev *CombEnv) Name() string { return ev.Nm }
func (ev *CombEnv) Desc() string { return ev.Dsc }

// Defaults sets initial default params
func (ev *CombEnv) Defaults() {
	ev.RndPctOn = 0.2
	ev.RndMinDiff = 0.5
}

// Config sets up the environment
func (ev *CombEnv) Config(typ PatsType, test bool, patsz evec.Vec2i, npools, ntrain, ntest, npats int) {
	ev.PatsType = typ
	ev.Test = test
	ev.PatsSize = patsz
	ev.NPools = npools
	ev.NTrain = ntrain
	ev.NTest = ntrain
	ev.NPats = npats

	ev.ConfigPats()
	ev.ConfigItems()

	if ev.Test {
		ev.Order = rand.Perm(ev.NTest)
		ev.Trial.Max = ev.NTest
	} else {
		ev.Order = rand.Perm(ev.NTrain)
		ev.Trial.Max = ev.NTrain
	}

	ev.Input.SetShape([]int{1, ev.NPools, ev.PatsSize.Y, ev.PatsSize.X}, nil, []string{"1", "Pool", "Y", "X"})
	if ev.PatsType == LinesToID {
		ev.Output.SetShape([]int{1, ev.NPools, 2, ev.PatsSize.X}, nil, []string{"1", "Pool", "HV", "X"})
	} else {
		ev.Output.SetShape([]int{1, ev.NPools, ev.PatsSize.Y, ev.PatsSize.X}, nil, []string{"1", "Pool", "Y", "X"})
	}
}

// ConfigItems configures the Train / Test lists
func (ev *CombEnv) ConfigItems() {
	np := ev.Pats.Rows
	ev.TrainItems.SetShape([]int{ev.NTrain, ev.NPools}, nil, []string{"itm", "pool"})
	ev.TestItems.SetShape([]int{ev.NTest, ev.NPools}, nil, []string{"itm", "pool"})

	rfnm := fmt.Sprintf("items_pats%d_pools%d_train%d_test%d_trn.tsv", np, ev.NPools, ev.NTrain, ev.NTest)
	sfnm := fmt.Sprintf("items_pats%d_pools%d_train%d_test%d_tst.tsv", np, ev.NPools, ev.NTrain, ev.NTest)
	_, err := os.Stat(rfnm)
	if !os.IsNotExist(err) {
		etensor.OpenCSV(&ev.TrainItems, gi.FileName(rfnm), '\t')
		etensor.OpenCSV(&ev.TestItems, gi.FileName(sfnm), '\t')
		return
	}

	for it := 0; it < ev.NTrain; it++ {
		for {
			for p := 0; p < ev.NPools; p++ {
				ev.TrainItems.Set([]int{it, p}, rand.Intn(np))
			}
			si := it * ev.NPools
			pv := ev.TrainItems.Values[si : si+ev.NPools]
			dupe := false
			for i := 0; i < it; i++ {
				oi := i * ev.NPools
				op := ev.TrainItems.Values[oi : oi+ev.NPools]
				if etensor.EqualInts(op, pv) {
					dupe = true
					break
				}
			}
			if !dupe {
				break
			}
		}
	}
	for it := 0; it < ev.NTest; it++ {
		for {
			for p := 0; p < ev.NPools; p++ {
				ev.TestItems.Set([]int{it, p}, rand.Intn(np))
			}
			si := it * ev.NPools
			pv := ev.TestItems.Values[si : si+ev.NPools]
			dupe := false
			for i := 0; i < it; i++ {
				oi := i * ev.NPools
				op := ev.TestItems.Values[oi : oi+ev.NPools]
				if etensor.EqualInts(op, pv) {
					dupe = true
					break
				}
			}
			if dupe {
				continue
			}
			for i := 0; i < ev.NTrain; i++ {
				oi := i * ev.NPools
				op := ev.TrainItems.Values[oi : oi+ev.NPools]
				if etensor.EqualInts(op, pv) {
					dupe = true
					break
				}
			}
			if !dupe {
				break
			}
		}
	}
	etensor.SaveCSV(&ev.TrainItems, gi.FileName(rfnm), '\t')
	etensor.SaveCSV(&ev.TestItems, gi.FileName(sfnm), '\t')
}

// ConfigPats configures the patterns
func (ev *CombEnv) ConfigPats() {
	switch ev.PatsType {
	case LinesToID:
		fallthrough
	case LinesToLines:
		nl := ev.PatsSize.X
		nhv := nl * 2
		ev.NPats = (nhv * (nhv - 1)) / 2
		sch := etable.Schema{
			{"Name", etensor.STRING, nil, nil},
			{"Input", etensor.FLOAT32, []int{ev.PatsSize.Y, ev.PatsSize.X}, []string{"Y", "X"}},
		}
		if ev.PatsType == LinesToID {
			sch = append(sch, etable.Column{"Output", etensor.FLOAT32, []int{2, ev.PatsSize.X}, []string{"Ln", "HV"}})
		} else {
			sch = append(sch, etable.Column{"Output", etensor.FLOAT32, []int{ev.PatsSize.Y, ev.PatsSize.X}, []string{"Y", "X"}})
		}
		ev.Pats.SetFromSchema(sch, ev.NPats)
		hv := [2]string{"H", "V"}
		row := 0
		for l1 := 0; l1 < nhv; l1++ {
			l1p := l1 % nl
			l1v := l1 / nl
			for l2 := l1 + 1; l2 < nhv; l2++ {
				l2p := l2 % nl
				l2v := l2 / nl
				in := ev.Pats.Col(1).(*etensor.Float32)
				out := ev.Pats.Col(2).(*etensor.Float32)
				ev.DrawLine(in, row, l1p, l1v == 1)
				ev.DrawLine(in, row, l2p, l2v == 1)
				if ev.PatsType == LinesToID {
					out.Set([]int{row, l1v, l1p}, 1)
					out.Set([]int{row, l2v, l2p}, 1)
				} else {
					ev.DrawLine(out, row, l1p, l1v == 1)
					ev.DrawLine(out, row, l2p, l2v == 1)
				}
				nm := fmt.Sprintf("%s%d%s%d", hv[l1v], l1p, hv[l2v], l2p)
				ev.Pats.SetCellString("Name", row, nm)
				row++
			}
		}
	case Random:
		np := ev.PatsSize.X * ev.PatsSize.Y
		nOn := patgen.NFmPct(ev.RndPctOn, np)
		minDiff := patgen.NFmPct(ev.RndMinDiff, np)
		sch := etable.Schema{
			{"Name", etensor.STRING, nil, nil},
			{"Input", etensor.FLOAT32, []int{ev.PatsSize.Y, ev.PatsSize.X}, []string{"Y", "X"}},
			{"Output", etensor.FLOAT32, []int{ev.PatsSize.Y, ev.PatsSize.X}, []string{"Y", "X"}},
		}
		ev.Pats.SetFromSchema(sch, ev.NPats)
		fnm := fmt.Sprintf("rndpats_%dx%d_n%d_on%d_df%d.tsv", ev.PatsSize.X, ev.PatsSize.Y, ev.NPats, nOn, minDiff)
		_, err := os.Stat(fnm)
		if !os.IsNotExist(err) {
			ev.Pats.OpenCSV(gi.FileName(fnm), etable.Tab)
		} else {
			in := ev.Pats.Col(1).(*etensor.Float32)
			out := ev.Pats.Col(2).(*etensor.Float32)
			patgen.PermutedBinaryMinDiff(in, nOn, 1, 0, minDiff)
			patgen.PermutedBinaryMinDiff(out, nOn, 1, 0, minDiff)
			for i := 0; i < ev.NPats; i++ {
				nm := fmt.Sprintf("P%03d", i)
				ev.Pats.SetCellString("Name", i, nm)
			}
			ev.Pats.SaveCSV(gi.FileName(fnm), etable.Tab, etable.Headers)
		}
	}
}

// DrawLine draws one line
func (ev *CombEnv) DrawLine(tsr *etensor.Float32, row, pos int, vert bool) {
	if vert {
		for y := 0; y < ev.PatsSize.Y; y++ {
			tsr.Set([]int{row, y, pos}, 1)
		}
	} else {
		for x := 0; x < ev.PatsSize.X; x++ {
			tsr.Set([]int{row, pos, x}, 1)
		}
	}
}

func (ev *CombEnv) Validate() error {
	return nil
}

func (ev *CombEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "Output":
		return &ev.Output
	}
	return nil
}

// ItemName returns name of item
func (ev *CombEnv) ItemName(itms *etensor.Int, row int) string {
	nm := ""
	si := row * ev.NPools
	for p := 0; p < ev.NPools; p++ {
		pi := itms.Values[si+p]
		inm := ev.Pats.CellString("Name", pi)
		nm += inm
		if p < ev.NPools-1 {
			nm += "_"
		}
	}
	return nm
}

// String returns the current state as a string
func (ev *CombEnv) String() string {
	row := 0
	if ev.Trial.Cur >= 0 {
		row = ev.Order[ev.Trial.Cur]
	}
	if ev.Test {
		return ev.ItemName(&ev.TestItems, row)
	} else {
		return ev.ItemName(&ev.TrainItems, row)
	}
}

// Init is called to restart environment
func (ev *CombEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

// CopyPat
func (ev *CombEnv) CopyPat(to, fm *etensor.Float32, pi int) {
	ps := fm.Len()
	si := pi * ps
	for i := 0; i < ps; i++ {
		to.Values[si+i] = fm.Values[i]
	}
}

// RenderItem sets the item
func (ev *CombEnv) RenderItem(itms *etensor.Int, row int) {
	si := row * ev.NPools
	for p := 0; p < ev.NPools; p++ {
		pi := itms.Values[si+p]
		ip := ev.Pats.CellTensorIdx(1, pi).(*etensor.Float32)
		op := ev.Pats.CellTensorIdx(2, pi).(*etensor.Float32)
		ev.CopyPat(&ev.Input, ip, p)
		ev.CopyPat(&ev.Output, op, p)
	}
}

// RenderState renders the state
func (ev *CombEnv) RenderState() {
	row := 0
	if ev.Trial.Cur >= 0 {
		row = ev.Order[ev.Trial.Cur]
	}
	if ev.Test {
		ev.RenderItem(&ev.TestItems, row)
	} else {
		ev.RenderItem(&ev.TrainItems, row)
	}
}

// Step is called to advance the environment state
func (ev *CombEnv) Step() bool {
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() { // true if wraps around Max back to 0
		erand.PermuteInts(ev.Order)
		ev.Epoch.Incr()
	}
	ev.RenderState()
	return true
}

func (ev *CombEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *CombEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*CombEnv)(nil)
