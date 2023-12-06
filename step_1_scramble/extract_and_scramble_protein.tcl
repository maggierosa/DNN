## This script goes through a trajectory and scramble the position and orientation of the protein
## Things that need to be modified and personalized: 
##   The definition of "atom1" and "atom2": the choice of these residues are not important. Any atom can do the job, as long as you choce at least 1 atom
##   For the valification test of saliency obtained in step_4, you may keep only a chosen partial of the protein in the last lines when saving the scrambled trajectory


source ./ball_sample.tcl

set psf [lindex $argv 0]
set traj [lindex $argv 1]
set out_psf [lindex $argv 2]
set out_dcd [lindex $argv 3]

echo "PSF is $psf"
echo "TRAJ is $traj"

mol load psf $psf
animate read dcd $traj waitfor all

set nf [molinfo top get numframes]

set ff [open "${out_dcd}.dat" w]

for {set fr 0} {$fr<$nf} {incr fr} {
    set results []
    set results2 []
    set prot [atomselect top "protein" frame $fr] 
    # random sample in sphere
    set com_sampled [sample_sphere 45.0]
    set results [concat [lindex $com_sampled 0] [lindex $com_sampled 1] [lindex $com_sampled 2] ]
    # random sample on sphere
    set vec [vecnorm [sample_sphere 1.0]] 
    set results2 [concat [lindex $vec 0] [lindex $vec 1] [lindex $vec 2] ]

    # Center protein 
    $prot moveby [vecinvert [measure center $prot]]
    
    ## Pick axis along two atoms then move it to randomly sampled sphere vector ($vec) 
    set atom1 [lindex [[atomselect top "resid 203 to 206 and name CA" frame $fr] get {x y z}] 0] # the choice of these residues are not important, any atom can do the job
    set atom2 [lindex [[atomselect top "resid 214 to 217 and name CA" frame $fr] get {x y z}] 0] # the choice of these residues are not important, any atom can do the job
    # Define axis
    set I [vecnorm [vecsub $atom2 $atom1]]
    # Move I to X axis
    $prot move [transvecinv $I]
    # Move X axis to randomly sampled vector $vec
    $prot move [transvec $vec]
    
    # Move COM of protein to randomly sampled sphere coordinate ($com_sampled) 
    $prot moveby $com_sampled
 
    puts $ff "$results $results2"
}


animate write psf $out_psf beg 0 end 0 skip 1 waitfor -1 sel [atomselect top "protein"] 0
animate write dcd $out_dcd beg 0 end -1 skip 1 waitfor -1 sel [atomselect top "protein"] 0

close $ff
exit

