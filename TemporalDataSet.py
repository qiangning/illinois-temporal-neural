import xml.etree.ElementTree as ET

class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        for d in self.data:
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            # self.token.append(d[:-(len(tmp[-1])+len(tmp[-2])+2)])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])
class temprel_set:
    def __init__(self, xmlfname):
        self.xmlfname = xmlfname
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        for e in root:
            self.temprel_ee.append(temprel_ee(e))
