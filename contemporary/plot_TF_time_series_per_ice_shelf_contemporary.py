import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob

fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(21.0,14.0))
axs = axs.ravel()

# Joughin 2021:
yr_PIG = [ 1994, 2007, 2009, 2010, 2012, 2014 ]
up_PIG = [ 2.01, 2.17, 2.28, 2.33, 1.61, 2.08 ]
mn_PIG = [ 1.81, 2.04, 2.11, 2.22, 1.43, 1.91 ]
lo_PIG = [ 1.61, 1.91, 1.94, 2.11, 1.25, 1.74 ]
yr_DOT = [ 2000, 2006, 2007, 2009, 2011, 2012, 2014, 2016 ]
up_DOT = [ 1.21, 1.84, 1.80, 2.04, 1.81, 1.30, 1.12, 1.10 ]
mn_DOT = [ 1.11, 1.66, 1.60, 1.84, 1.66, 1.16, 1.04, 1.04 ]
lo_DOT = [ 1.01, 1.48, 1.40, 1.64, 1.51, 1.02, 0.96, 0.98 ]

bm = xr.open_dataset('/data/njourdain/DATA_ISMIP6/BedMachineAntarctica_2020-07-15_v02_8km.nc')

msk = xr.open_dataset('/data/njourdain/DATA_ISMIP6/Mask_Iceshelf_IMBIE_8km.nc')

print(msk.NAME.values)

colors = [ 'cornflowerblue', 'darkblue'  , 'yellow'    , 'brown'      ,  'cyan'     ,   'magenta'   ,   'chartreuse'  ,    'green'   ,    'red'   ,   'orange'   , 'lightpink'    , 'olive'       , 'gold'     , 'lavender' , 'skyblue'    ]

isf = [ 'Pine_Island', 'Dotson', 'Totten', 'Ronne' ]
#isf = [ 'Pine_Island', 'Getz', 'Totten', 'Ronne' ]

years = np.arange(1950,2026,1)

alph = ['(a) ', '(b) ', '(c) ', '(d) ' ]

for kisf in range(len(isf)):

   if ( isf[kisf] == 'Dotson' ):
      namisf = 'Dotson/Philbin_Inlet'
   else:
      namisf = isf[kisf]
   idx = np.where( ( msk.NAME.values == namisf ) )[0][0]
   print(isf[kisf], idx)

   mask = msk.Iceshelf.where( ((msk.Iceshelf == idx+1)&(bm.mask == 1)), 0 )/(idx+1)

   aa = mask.sum(dim=["x","y"]).values
   print('   ',aa*8*8,' km2')

   for scenar in [ 'main', 'cold', 'warm', 'vary' ]:

      file_in = 'tf_ISdraft_Oyr_contemporary_'+scenar+'_ismip8km_60m_1950-2025.nc'
      ds = xr.open_dataset(file_in)

      tmp = ds.tf * mask
      mean_TF = tmp.sum(dim=["x","y"]).values / aa

      print('   ',scenar,mean_TF[0],mean_TF[-1])

      if scenar == 'main':
        lab = 'Main plausible pathway'
        col = 'gray'
      elif scenar == 'cold':
        lab = 'Cold plausible pathway'
        col = 'cornflowerblue'
      elif scenar == 'warm':
        lab = 'Warm plausible pathway'
        col = 'firebrick'
      elif scenar == 'vary':
        lab = 'Varying plausible pathway'
        col = 'orange'
      axs[kisf].plot(years,mean_TF,color=col,linewidth=1.5,label=lab)

   tit = alph[kisf]+isf[kisf]
   axs[kisf].set_title(tit,fontsize=20,fontweight='bold')
   axs[kisf].set_ylabel('Mean Thermal Forcing (°C)',fontsize=18)
   axs[kisf].set_xlim([1950,2025])
   axs[kisf].set_ylim([0,3.5])
   axs[kisf].tick_params(axis='both', which='both', labelsize=16)

   if ( isf[kisf] == 'Dotson' ):
      for kyr in range(len(yr_DOT)):
         axs[kisf].plot([yr_DOT[kyr],yr_DOT[kyr]],[lo_DOT[kyr],up_DOT[kyr]],'k',linewidth=2)
         axs[kisf].scatter(yr_DOT[kyr],mn_DOT[kyr],s=30,c='k',edgecolor='none')
         # legend:
         axs[kisf].plot([1970,1970],[0.4,0.6],'k',linewidth=2)
         axs[kisf].scatter(1970,0.5,s=30,c='k',edgecolor='none')
         axs[kisf].text(1972,0.5,'Observational estimates',fontsize=16,va='center')
   elif ( isf[kisf] == 'Pine_Island' ):
      for kyr in range(len(yr_PIG)):
         axs[kisf].plot([yr_PIG[kyr],yr_PIG[kyr]],[lo_PIG[kyr],up_PIG[kyr]],'k',linewidth=2)
         axs[kisf].scatter(yr_PIG[kyr],mn_PIG[kyr],s=30,c='k',edgecolor='none')
         # legend:
         axs[kisf].plot([1970,1970],[0.4,0.6],'k',linewidth=2)
         axs[kisf].scatter(1970,0.5,s=30,c='k',edgecolor='none')
         axs[kisf].text(1972,0.5,'Observational estimates',fontsize=16,va='center')

axs[3].legend(loc='upper center',fontsize=16)

figname='TF_timeseries_contemporary.pdf'

fig.savefig(figname)
