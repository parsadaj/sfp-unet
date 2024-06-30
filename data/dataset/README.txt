Deep Shape From Polarization: Dataset

Enclosed is the data used for all experiments in our paper (https://arxiv.org/pdf/1903.10210.pdf).
Data was collected by the team at PKU, and prepared by the team at UCLA. We hope
that this dataset can help promote further exploration into the exciting task of
shape from polarization.

Data Summary:

    In total, the dataset contains 33 objects. 25 of the objects are reserved for
    the train set used in the paper, while the remaining 8 are used for the test
    set (see enclosed train_list.csv and test_list.csv for the exact train and test
    splits respectively).

    For the majority of objects (those with asymmetries for different views),
    data was captured with the object in 4 different orientations:
    front (f), back (b), left (l), and right (r).

    Additionally, each of these was repeated under 3 different lighting conditions:
    indoor lighting, outdoor on a sunny day, and outdoor on a cloudy day. After
    curating the set to remove similarities between the train/test splits, the
    final item counts for the two splits are 237/27.

Data Format:

    Each data item is stored in a single Matlab file, as a dictionary with 3 entries:

        1 - images [1024x1224x4 double] = The polarization images, where each channel
        corresponds to a different angle of the polarizer: 0°, 45°, 90°, and 135°.

        2 - mask [1024x1224 uint8] = The foreground mask for each object.

        3 - Normals_gt [1024x1224x3 double] = The ground truth surface normals for
        each object. The channels correspond to X, Y, and Z components of the normals
        at each pixel.

    Naming Convention: <lighting_condition>/<item_name>_<orientation>.mat

        Each sub-directory corresponds to a different lighting condition (i.e. indoor,
        outdoor_sunny, or outdoor_cloudy). The orientation is one of f, b, l, or r,
        or missing for objects which were only captured in a single orientation due
        to symmetry.

Collection Methods:

    Polarization images were captured using a Phoenix 5.0 MP Polarization camera
    from Lucid Vision Labs (https://thinklucid.com/product/phoenix-5-0-mp-polarized-model/).

    Ground truth surface normals were obtained through a 3 step process:

        1 - Obtain high-quality 3D shapes using a SHINING 3D scanner
        (https://www.einscan.com/desktop-3d-scanners/einscan-sp/), with single shot
        accuracy no more than 0.1mm, point distance from 0.17 mm to 0.2 mm, and a
        synchronized turntable for automatically registering scanning from multiple
        viewpoints.

        2 - Align scanned 3D mesh's from the scanner’s coordinate system to the
        image coordinate system of the polarization camera using 3D modeling tool
        MeshLab (https://www.meshlab.net/).

        3 — Compute surface normals using the Mitsuba renderer
        (http://www.mitsuba-renderer.org/).
