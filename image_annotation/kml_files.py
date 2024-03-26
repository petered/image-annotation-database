import os.path
from typing import Sequence, Tuple
import xml.dom.minidom as md


def write_kml_file(path: str, name: str, latlng_coords: Sequence[Tuple[float, float]]):
    # Create the KML document
    kml_doc = md.Document()

    # Create the root KML element
    kml_element = kml_doc.createElement("kml")
    kml_element.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
    kml_doc.appendChild(kml_element)

    # Create the Document element
    document_element = kml_doc.createElement("Document")
    kml_element.appendChild(document_element)

    # Create the Placemark element
    placemark_element = kml_doc.createElement("Placemark")
    document_element.appendChild(placemark_element)

    # Create and set the name element
    name_element = kml_doc.createElement(name)
    name_text = kml_doc.createTextNode("Path")
    name_element.appendChild(name_text)
    placemark_element.appendChild(name_element)

    # Create and set the LineString element
    line_string_element = kml_doc.createElement("LineString")
    placemark_element.appendChild(line_string_element)

    # Create and set the coordinates element
    coordinates_element = kml_doc.createElement("coordinates")
    coordinates_text = kml_doc.createTextNode(" ".join(f"{coord[1]},{coord[0]},0" for coord in latlng_coords))
    coordinates_element.appendChild(coordinates_text)
    line_string_element.appendChild(coordinates_element)

    # Write the KML file
    path = os.path.expanduser(path)
    with open(path, "w") as kml_file:
        kml_file.write(kml_doc.toprettyxml(indent="  "))
    print(f"Wrote {path}")