set psfname [lindex $argv 0]
set dcdname [lindex $argv 1]
set firstframe [lindex $argv 2]
set template [lindex $argv 3]
set outdcdname [lindex $argv 4]
package require pbctools

## This script goes through a trajectory of simulation to obtain the protein and align them with a template
## Things that need to be modified and personalized: 
##   the way to align the protein

set traj [mol load psf $psfname]
mol addfile ${dcdname} waitfor all

if {$firstframe>0} {
set discardtill [expr $firstframe - 1]
animate delete beg 0 end $discardtill
}

pbc wrap -all -compound res -center bb -centersel "protein"

set framecount [molinfo top get numframes]
set tmpl [mol load pdb $template]
for {set i 0} {$i<$framecount } {incr i} {
set trajprot [atomselect $traj "name CA and resid 211 to 220 70 to 76 79 to 86 103 to 108 113 to 118 131 to 140 146 to 150 170 to 175 184 to 189" frame $i]
set tmplprot [atomselect $tmpl "name CA and resid 211 to 220 70 to 76 79 to 86 103 to 108 113 to 118 131 to 140 146 to 150 170 to 175 184 to 189" frame $i]
set trajall [atomselect $traj all frame $i]
set transmat [measure fit $trajprot $tmplprot]
$trajall move $transmat 
$trajall delete
$tmplprot delete
$trajprot delete
}
mol delete $tmpl

set sel [atomselect $traj "protein"]
$sel writepsf ${outdcdname}.psf
animate write dcd ${outdcdname}.dcd sel $sel $traj
$sel delete

mol delete all
exit
