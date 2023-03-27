source ./ball_sample.tcl

set psf [lindex $argv 0]
set traj [lindex $argv 1]
set out_psf [lindex $argv 2]
set out_dcd [lindex $argv 3]

echo "PSF is $psf"
echo "TRAJ is $traj"

mol load psf $psf
animate read xtc $traj beg 0 end -1 waitfor all

set nf [molinfo top get numframes]

set ff [open "${traj}.dat" w]

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
    set atom1 [lindex [[atomselect top "index 4437" frame $fr] get {x y z}] 0]
    set atom2 [lindex [[atomselect top "index 1110" frame $fr] get {x y z}] 0]
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

animate write psf $out_psf beg 0 end 0 skip 1 waitfor -1 sel [atomselect top "protein and resid 39 to 68 75 to 101 110 to 129 137 to 167 171 to 202 233 to 263 292 to 318 328 to 353 358 to 374 382 to 416 424 to 450 465 to 491"] 0

animate write dcd $out_dcd beg 0 end -1 skip 1 waitfor -1 sel [atomselect top "protein and resid 39 to 68 75 to 101 110 to 129 137 to 167 171 to 202 233 to 263 292 to 318 328 to 353 358 to 374 382 to 416 424 to 450 465 to 491"] 0

close $ff
exit

