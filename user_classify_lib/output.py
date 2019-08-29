"""
Customized library to format the results from CorEx and generate graphs based on the results.

Author: Eric Hu

"""
import numpy as np

class LatexClusterChartGenerator():

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w")

    # region Header
    def writeHeader(self):
        self.f.write(
            """% \\BEGIN_FOLD and END_FOLD used by TeXstudio but have no effect in Overleaf

\\documentclass[class=minimal, border=8pt]{standalone}

% Packages
\\usepackage[utf8]{inputenc}
\\usepackage[dvipsnames]{xcolor}

\\usepackage{tikz}
\\usetikzlibrary{positioning, backgrounds, fit, calc}

%BEGIN_FOLD Set up Constants
\\newcommand{\\innerSep}{0.15cm} % Text buffer around text in nodes and around nodes in surround
\\newcommand{\\columnXShift}{5cm + 3cm + 2 * \\innerSep} % Column title width plus separation plus buffer
\\newcommand{\\dashOpacity}{0.0} % Set greater than zero to see dashed guidelines
%END_FOLD

%BEGIN_FOLD Set up Colors
\\definecolor{Layer1ItemOutline}{rgb}{0.25, 0.75, 0.25}
\\definecolor{Layer1ItemFill}{rgb}{0.75, 1.0, 0.75}
\\definecolor{Layer1ItemOutlineSuppressed}{rgb}{0.5, 1.0, 0.5}
\\definecolor{Layer1ItemFillSuppressed}{rgb}{0.9, 1.0, 0.9}
\\definecolor{Layer1GroupOutline}{rgb}{0.0, 0.25, 0.0}
\\definecolor{Layer1GroupFill}{rgb}{0.95, 1.0, 0.95}

\\definecolor{Layer2ItemOutline}{rgb}{0.25, 0.25, 0.75}
\\definecolor{Layer2ItemFill}{rgb}{0.75, 0.75, 1.0}
\\definecolor{Layer2ItemOutlineSuppressed}{rgb}{0.5, 0.5, 1.0}
\\definecolor{Layer2ItemFillSuppressed}{rgb}{0.9, 0.9, 1.0}
\\definecolor{Layer2GroupOutline}{rgb}{0.0, 0.0, 0.25}
\\definecolor{Layer2GroupFill}{rgb}{0.95, 0.95, 1.0}
%END_FOLD

%BEGIN_FOLD Set up TikZ styles
% Titles
\\tikzstyle{documentTitle} = [
	align=center,
	font=\\fontsize{16}{0}\\bfseries\\selectfont
]
\\tikzstyle{columnTitle} = [
    text width=5cm,
    align=left,
    font=\\bfseries, % Bold
    inner sep=\\innerSep,
    rounded corners,
    dashed,
    draw=black,
    draw opacity=\\dashOpacity
]
\\tikzstyle{groupTitle} = [
    text width=8cm,
    align=left,
    font=\\itshape, % Italics
    inner sep=\\innerSep,
    rounded corners,
    dashed,
    draw=black,
    draw opacity=\\dashOpacity
]
% Visible Items
\\tikzstyle{Layer1Item} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer1ItemOutline,
    fill=Layer1ItemFill
]
\\tikzstyle{Layer2Item} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer2ItemOutline,
    fill=Layer2ItemFill
]
% Suppressed Items
\\tikzstyle{Layer1ItemSuppressed} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer1ItemOutlineSuppressed,
    fill=Layer1ItemFillSuppressed,
    text=textSuppressed
]
\\tikzstyle{Layer2ItemSuppressed} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer2ItemOutlineSuppressed,
    fill=Layer2ItemFillSuppressed,
    text=textSuppressed
]
% Surrounds
\\tikzstyle{Layer1GroupSurround} = [
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer1GroupOutline,
    fill=Layer1GroupFill
]
\\tikzstyle{Layer2GroupSurround} = [
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer2GroupOutline,
    fill=Layer2GroupFill
]
% Arrows
\\tikzstyle{arrow} = [
	thick,
	->,
	>=stealth,
	rounded corners = 5pt
]
\\tikzstyle{arrowSuppressed} = [
	thick,
	->,
	>=stealth,
	opacity=0.1,
	rounded corners = 5pt
]
\\tikzstyle{arrowHidden} = [
    opacity=0
]
%END_FOLD
% Begin document
\\begin{document}

% Begin TikZ flow-chart environment
\\begin{tikzpicture}[
    node distance=1cm % Used as default vertical spacing between adjacent nodes
]

% Title
\\node (documentTitleNode) [documentTitle, anchor=center] at (2.5, 1) {Clustering Results from CorEx Analysis};"""
        )

    # endregion

    # region Footer
    def writeFooter(self):
        self.f.write(
            """
            % End TikZ flow-chart environment
\\end{tikzpicture}

% End document
\\end{document}
            """
        )
        self.f.close()
    # endregion

    #region Clusters

    def writeClusters(self, clusters: np.ndarray, n_clusters: int, layer: int = 0, cluster_order=None, tc : float = None):
        tc_value = "N/A"
        if tc is not None:
            tc_value = str(tc)

        if layer == 1:
            self.f.write("""\n%BEGIN_FOLD Column Nodes: Layer {0}
\\node (Layer{0}ColumnTitle) [columnTitle] at (0, 0) {{Layer {0}, Total TC: {1}}};""".format(layer, tc_value))
        else:
            self.f.write("""\n%BEGIN_FOLD Column Nodes: Layer {0}
\\node (Layer{0}ColumnTitle) [columnTitle, right=\\columnXShift*80 of Layer{1}ColumnTitle.center, anchor=center] {{Layer {0}, Total TC: {2}}};""".format(layer, str(layer-1), tc_value))

        previous_cluster = "Layer{0}ColumnTitle".format(layer)

        for j in range(n_clusters):

            if cluster_order is None:
                c_num = j
            else:
                c_num = cluster_order[j]
            cluster_name = "Layer{0}GroupCluster{1}".format(layer, c_num)
            cluster_title = "Cluster {0}".format(c_num)
            self.f.write("\n% {0}".format(cluster_title))
            if j == 0:
                self.f.write("\n\\node ({0}) [groupTitle,below=of {1}.south west, anchor=west] {{{2}}};".format(cluster_name, previous_cluster, cluster_title))
            else:
                self.f.write(
                    "\n\\node ({0}) [groupTitle,below=of {1}.south west, anchor=west, xshift=\\innerSep] {{{2}}};".format(cluster_name,
                                                                                                       previous_cluster,
                                                                                                       cluster_title))
            self.f.write("\n\\foreach \\name/\\type/\\below/\\text in {")
            previous_node = cluster_name
            cluster_header = "Layer{0}".format(layer)
            for k in range(len(clusters[c_num])):
                if k == 0:
                    self.f.write("\n\t{0}{1}/Layer{2}Item/{3}/{4}".format(cluster_header, str(clusters[c_num][k]).replace(" ", "_").replace("|", "_"), layer, previous_node, str(clusters[c_num][k]).replace("_", " ")))
                else:
                    self.f.write(",\n\t{0}{1}/Layer{2}Item/{3}/{4}".format(cluster_header, str(clusters[c_num][k]).replace(" ", "_").replace("|", "_"), layer,
                                                                          previous_node,
                                                                          str(clusters[c_num][k]).replace("_", " ")))
                previous_node = "{0}{1}".format(cluster_header, str(clusters[c_num][k].replace(" ", "_").replace("|", "_")))
            self.f.write("""}}
{{
	\\node (\\name) [\\type, below of=\\below] {{\\text}};
}}
\\begin{{scope}}[on background layer]
	\\node ({0}Surround) [Layer{1}GroupSurround, fit = ({0}) ({2})] {{}};
\end{{scope}}
            
            """.format(cluster_name, layer, previous_node))
            previous_cluster = cluster_name + "Surround"

        self.f.write("%END_FOLD")



    #endregion

    #region Arrows
    def writeArrows(self, source_n_clusters, source_layer_name, source_layer=1, dest_layer = 2):
        self.f.write("\n%BEGIN_FOLD Arrows")
        self.f.write("\n\\foreach \\type/\start/\\finish[count=\\count from 0] in {")
        for i in range(source_n_clusters):
            if i == 0:
                self.f.write("\n\tarrow/Layer{0}GroupCluster{1}Surround/Layer{2}{3}".format(source_layer, i, dest_layer, source_layer_name + str(i)))
            else:
                self.f.write(",\n\tarrow/Layer{0}GroupCluster{1}Surround/Layer{2}{3}".format(source_layer, i, dest_layer,
                                                                                            source_layer_name + str(
                                                                                                i)))
        self.f.write("""}
{
	\\draw [\\type]
		(\start.east) --
		(\\finish.west);
}
%END_FOLD""")

    #endregion


class LatexFeaturesBarGraphGenerator():

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w")

    # region Header
    def writeHeader(self, title: str):
        self.f.write(
            """\\documentclass{{article}}
\\usepackage{{pgfplots}}

\\usepackage{{fancyhdr}}
\\fancyhf{{}} % clear all header and footers
\\renewcommand{{\headrulewidth}}{{0pt}} % remove the header rule
\\pagestyle{{fancy}}

\\pgfplotsset{{compat=1.9}}
\\title{{\\sffamily Graphical Representation of {0}}}
\\date{{}}
\\begin{{document}}
\\maketitle
\\begin{{tikzpicture}}""".format(title.replace('_', ' ').replace(',', '{,}'))
        )

    # endregion

    # region Footer
    def writeFooter(self):
        self.f.write(
            """
\\end{tikzpicture}

\\end{document}"""
        )
        self.f.close()

    # endregion

    #region Formatting the Stacked Bar Graph
    def formatStackedGraph(self):
        self.f.write(
            """            
            every axis title/.append style={font=\large\sffamily,color=black!60},
            xbar stacked,
            width=12cm, height=3.5cm, enlarge y limits=1.5,
            xmin=0,
            ticks=none,
            axis line style={draw=none},
            symbolic y coords={response},
            ytick=data,
            legend style={at={(0.5,1.05)}, draw=none, font=\sffamily,
                anchor=north,legend columns=-1},
            nodes near coords,
            nodes near coords align={below},
            every node near coord/.append style={
         yshift=-3pt}]"""
        )
    #endregion

    #region Formatting the Bar Graph
    def formatBarGraph(self):
        self.f.write(
            """            every axis title/.append style={font=\\large\\sffamily,color=black!60},
            xbar,
            xmin=0,
            ytick=data,
            y tick label style={anchor=east, align=right,text width=5cm},
            nodes near coords
         ]"""
        )
    #endregion

    def pageBreak(self):
        self.f.write("""
    \\end{tikzpicture}
    \\newpage
    \\begin{tikzpicture}
        """)

    #region Generating the Stacked Bar Graph
    def makeFeaturesStackedGraph(self, offset:int, title:str, response_names:list, response_values:list):
        self.f.write("\n\t\\begin{{scope}}[yshift={0}cm]".format(offset))
        self.f.write("\n\t\t\\begin{axis}[")
        self.f.write("\n\t\t\ttitle=\\textbf{{{0}}},".format(title.replace('_', ' ').replace(',', '{,}')))
        self.formatStackedGraph()
        for val in response_values:
            self.f.write("\n\t\t\\addplot coordinates {{({0},response)}};".format(val))
        legend_str = "\n\t\t\\legend{"
        add_comma = False
        for name in response_names:
            if add_comma:
                legend_str = legend_str + ", "
            else:
                add_comma = True
            legend_str = legend_str + "\\strut " + str(name).replace('_', ' ').replace(',', '{,}')
        legend_str = legend_str + "}"
        self.f.write(legend_str)
        self.f.write("\n\t\t\\end{axis}")
        self.f.write("\n\t\\end{scope}")

    #endregion

    #region Generating the Stacked Bar Graph
    def makeFeaturesBarGraph(self, title:str, response_names:list, response_values:list):
        self.f.write("\n\t\\begin{scope}[yshift=0cm]")
        self.f.write("\n\t\t\\begin{axis}[")
        self.f.write("\n\t\t\ttitle=\\textbf{{{0}}},".format(title.replace('_', ' ').replace(',', '{,}')))
        self.f.write("\n\t\t\twidth=\\textwidth, height={0}cm, enlarge y limits={1},".format(len(response_names), 0.5/len(response_names)))
        label_string = "\n\t\t\tsymbolic y coords={"
        plotting_string = "\n\t\t\\addplot coordinates { "
        for i in range(len(response_names)):
            if i != 0:
                label_string = label_string + ', '
            label_string = label_string + str(response_names[i]).replace('_', ' ').replace(',', '{,}')
            plotting_string = plotting_string + '(' + str(response_values[i]) + ',' + str(response_names[i]).replace('_', ' ').replace(',', '{,}') + ')'

        label_string = label_string + "},"
        plotting_string = plotting_string + "};"

        self.f.write(label_string)
        self.formatBarGraph()
        self.f.write(plotting_string)

        self.f.write("\n\t\t\\end{axis}")
        self.f.write("\n\t\\end{scope}")

    #endregion