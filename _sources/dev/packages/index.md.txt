# Packages

Overview and deep dives into the code structure of `i7aof` packages. Each page outlines responsibilities, key functions/classes, data flow, and extension points.

```{toctree}
:maxdepth: 1
:caption: i7aof packages

biascorr
grid
imbie
io
remap
topo
vert
```

```{note}
Maintainers, when adding new packages, please keep them alphabetically listed
above.  Each page should follow a common structure:
Purpose → Public Python API → Required config options → Outputs → Data model →
Runtime and external requirements → Usage → Internals → Edge cases →
Extension points.
```