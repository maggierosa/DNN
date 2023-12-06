set psf [lindex $argv 0]
set traj [lindex $argv 1]
set out_psf [lindex $argv 2]
set out_dcd [lindex $argv 3]

echo "PSF is $psf"
echo "TRAJ is $traj"

mol load psf $psf
mol addfile $traj waitfor all

set nf [molinfo top get numframes]

for {set fr 0} {$fr<$nf} {incr fr} {
    set results []
    set results2 []
    set prot [atomselect top "protein" frame $fr] 
    set com [measure center $prot]
    set matrixx [transaxis x 180]
    set matrixy [transaxis y 180]
    $prot moveby [vecscale -1.0 $com]
    $prot move $matrixx
    $prot move $matrixy
    $prot moveby $com
}

animate write psf $out_psf beg 0 end 0 skip 1 waitfor -1 sel [atomselect top "protein"] 0

animate write dcd $out_dcd beg 0 end -1 skip 1 waitfor -1 sel [atomselect top "protein"] 0

exit

