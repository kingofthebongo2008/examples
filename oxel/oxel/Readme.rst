===========================================================================
About
===========================================================================
Oxel is a tool for generating conservative low polygon occluder proxies for 3D geometry that are cheap to render and can be used for occlusion culling in games.

http://www.nickdarnell.com/Oxel

The research for Oxel started soon after I began writing about Hierarchical Z-Buffer Occlusion Culling,

* `Hierarchical Z-Buffer Occlusion Culling <http://www.nickdarnell.com/2010/06/hierarchical-z-buffer-occlusion-culling>`_
* `Hierarchical Z-Buffer Occlusion Culling - Shadows <http://www.nickdarnell.com/2010/07/hierarchical-z-buffer-occlusion-culling-shadows>`_

After doing those articles I began working on a project that would generate occluders for geometry, which lead to this article.

* `Hierarchical Z-Buffer Occlusion Culling - Generating Occlusion Volumes <http://www.nickdarnell.com/2011/06/hierarchical-z-buffer-occlusion-culling-generating-occlusion-volumes>`_

But the initial method of voxelization was flawed it wouldn't work on real game art, so after some additional R&D I wrote this post on another more robust method of voxelization.

* `Robust Inside and Outside Solid Voxelization <http://www.nickdarnell.com/2011/09/robust-inside-and-outside-solid-voxelization>`_

Which finally lead to this post,

* `Oxel - A Tool for Occluder Generation <http://www.nickdarnell.com/2012/04/oxel-a-tool-for-occluder-generation>`_

===========================================================================
License
===========================================================================

The MIT License (MIT)

Copyright (c) 2012 Nick Darnell < NickDarnell@gmail.com >

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

===========================================================================
Latest Version (1.2.1)
===========================================================================

* The Winding Order option now works.

* Removed the option for simulated annealing, because it isn't ready for prime-time.

* Fixed a bug introduced in 1.2.0 that was preventing the settings file from ever being used in the GUI interface or the new console interface.  Both the console and GUI interfaces will now default to whatever the settings are in the file, and can be overridden using command line args.

===========================================================================
Version (1.2.0)
===========================================================================

* Feature: Added a command line interface, use -c to invoke it.  Use -help or -h to get a list of commands that you can use.  If you use the command line mode, the Oxel.Settings.xml (the settings set in the UI) won't be used.  You need to explictly provide all the settings you want, using the command line interface otherwise the defaults will be used.  The simplest way to use the command line interface is to call: Oxel.exe <InputMesh> <OutputMesh>.

* Feature: Improved the feedback given by the progress window.

* Feature: Moved the processing into a background thread.

* Feature: Added better support for adding new mesh importers.

* Feature: Added the new AlphaBlending mode to visualize both the original mesh and occluder at the same time.

* Bug Fix: Fixed a bug with the voxel field, if set too large it stopped working.  This is no longer the case, though processing a large voxel field still takes a long time.