from setuptools import find_packages, setup

package_name = "ares_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Patroclus Systems",
    maintainer_email="engineering@patroclus.systems",
    description="ARES perception. Stereo depth with confidence, and traversability.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "stereo_node = ares_perception.stereo_node:main",
            "speed_governor = ares_perception.governor_node:main",
            "traversability_node = ares_perception.traversability_node:main",
            "segmentation_node = ares_perception.segmentation_node:main",
            "detection_node = ares_perception.detection_node:main",
        ],
    },
)
