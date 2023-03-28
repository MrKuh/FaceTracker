## FAQ : Path.mkdir()

Creates a **new path** at the given location.

If **parents** is set to **true**, any missing parents of this path are created as needed.

If **parents** is set to **false**, and a directory is missing from the path then it raises an **error**.

- Like for example if the folder name kamal is missing from the path which we give to save the image, then that raises an error.

If **exist_ok** is set to true, then even if the folder already exists then it doesn't raise any errors and just tries to save the images in that folder.

If **exist_ok** is set to **false**, then if that folder already exists, then it will stop execution and will show an error.

---

Read more on the official page . [Know More](https://docs.python.org/3/library/pathlib.html#methods-and-properties)
