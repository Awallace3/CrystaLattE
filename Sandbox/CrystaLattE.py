#!/usr/bin/env python

import Read_CIF
import psi4
import re
import os
from psi4.driver.wrapper_autofrag import auto_fragments

def main():

    # ==================================================================
    # Read a CIF file and generates a supercell.
    print ("")
    print ("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ")
    print (" STEP 1. GENERATION OF A SUPERCELL FROM A CIF FILE.")
    print ("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ")
    print ("")
    
    ReadCifIn = 'Benzene-138K.cif'  # CIF input file. # WARNING: Hardcoded for now!
    ReadCifOut = 'Benzene-138K.xyz' # XYZ output file. # WARNING: Hardcoded for now!
    ReadCifA = '2'                  # X replicas. # WARNING: Hardcoded for now!
    ReadCifB = '2'                  # Y replicas. # WARNING: Hardcoded for now!
    ReadCifC = '2'                  # Z replicas. # WARNING: Hardcoded for now!
    
    args = ['', '-i', ReadCifIn, '-o', ReadCifOut, '-b', ReadCifA, ReadCifB, ReadCifC]
    
    print ("The following arguments will be passed to the CIF reader script:")
    print ("./Read_CIF.py" + " ".join(str(arg) for arg in args))
    print ("")
    
    print ("--------------------------------------------------------------------- ")
    Read_CIF.main(args)
    print ("--------------------------------------------------------------------- ")
    # ==================================================================
    
    # ==================================================================
    # Take the supercell .xyz file
    # And generate .xyz files with all possible monomers.
    print ("")
    print ("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ")
    print (" STEP 2. EXTRACTION OF MONOMERS FROM THE SUPERCELL.")
    print ("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ")
    print ("")
    
    # Read the lines in the .xyz file recently generated by Read_CIF.py
    with open(ReadCifOut) as fxyz:
        sxyz = fxyz.readlines()
        #print(sxyz[2:])
    
    # Generate a SuperCell object.
    SuperCell = psi4.geometry('\n'.join(sxyz[2:]))
    SuperCell.update_geometry()
    #print (SuperCell.print_out())
    
    # Generate fragments from SuperCell.
    #CellFrags = auto_fragments(molecule = SuperCell)
    #print (CellFrags.natom())
    #print (CellFrags.nfragments())
    
    # Read the output of the automatic fragmentation.
    p4frag = "bzfrag.p4" # WARNING: Name of the fragmented super cell
                         # file is temporarily hardcoded.

    with open(p4frag) as cellfrags:
            frags = cellfrags.readlines()

    # Generate .xyz files for each fragment.
    numfrags = 664 # WARNING: Number of fragments temporarily hardcoded!
    numfatoms = 12 # WARNING: Number of atoms per fragment hardcoded!
                   # WARNING: What if there are two types of molecules?
    frg_separator = "--" # Fragment separator string.
    fcounter = 1 # Counter of processed fragments.
    lcounter = 0 # Counter of lines.
    HeaderLine = True # Flag for first line of a fragment .xyz file.
    
    for line in frags:
        lcounter += 1
        
        if line.startswith(frg_separator):
            fcounter += 1
            HeaderLine = True
        
        else:
            ffnidx = "f" + str(fcounter).zfill(len(str(numfrags))) \
                     + ".xyz"
            with open(ffnidx, "a") as frgxyz: # WARNING: File exists?
                
                if HeaderLine == True:
                    frgxyz.write(str(numfatoms) + "\n"+ "Fragment " 
                    + str(fcounter) + "\n")
                    HeaderLine = False
                
                if HeaderLine == False:
                    frgxyz.write(line)
    
    # Discard fragments that are not a complete molecule.
    directory = os.getcwd()
    files = os.listdir(directory)

    print("")
    print("Detecting fragments with incomplete molecules:")
    print("")

    incmolcounter = 0

    for file in files:
        if re.match('^f[0-9]+.xyz$', file): # Match filenames f???.xyz
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) != numfatoms + 2:
                    print("Expected " + str(numfatoms)\
                          + " atoms. Found only " + str(len(lines) - 2)\
                          + ". Removing: " + file)
                    os.remove(os.path.join(directory,file))
                    incmolcounter += 1
    
    print ("Removed %s fragments." % incmolcounter)
    
    # Generate monomers out of fragments

    # ==================================================================
    
    # ==================================================================
    # Loop through all monomers and generate dimers with all other
    # monomers.
    #
    # Filter dimers that are too distant apart.
    #
    # Filter out and keep count of all non-unique dimers, using the
    # nuclear repulsion energy criteria.
    # ==================================================================
    
    # ==================================================================
    # Loop through all dimers and generate trimers with all other
    # monomers.
    #
    # Filter trimers that are too distant apart.
    #
    # Filter out and keep count of all non-unique trimers, using 
    # ArbAlign.
    # ==================================================================
    
    # .
    # .
    # .
    
    # ======================================================================
    # Run plesantly parallel PSI4 computations on all the final list of 
    # monomers, dimers, trimers, etc.
    #
    # Multiply the resulting energy of each one by the degeneracy factor.
    #
    # Sum results to get a lattice energy.
    # ======================================================================

if __name__ == "__main__":
    main()
