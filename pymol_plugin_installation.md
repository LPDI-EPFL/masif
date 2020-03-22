# Installing the masif pymol plugin. 




To install the plugin go to the Plugin -> Plugin Manager window in PyMOL and choose the Install new plugin tab:

![MaSIF Install new plugin window](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/ImageInitial.png)

Then select the masif/source/masif_pymol_plugin.zip file: 

![MaSIF Install new plugin window](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/PluginSelect.png)

After this, pymol will prompt you for an installation directory. You can select the default path. 

Finally, close and reopen pymol. Go again to the plugin manager window and verify that masif pymol plugin is installed: 


![MaSIF Install new plugin window](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/ImageVerify.png)


You can now test the installation of the plugin. For example, you can download any of the files in this link : 

https://github.com/LPDI-EPFL/masif/tree/master/comparison/masif_site/masif_vs_sppider/masif_pred

and then open them using the command (inside pymol):

```
loadply 4ETP_A.ply
```


## Troubleshooting the plugin installation.

- Make sure that pymol can find the location where you installed the plugin. Another possibilty is to go to "Plugin Manager" within PyMOL and then click the "Settings" tab. There, click on "Add new directory" add the following directory:

```
masif/source/masif_plugin_pymol/
```

This should tell pymol to search this directory for plugins.
