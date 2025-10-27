import streamlit as st
import numpy as np
import sympy as sp
from sympy import primefactors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
import datetime

# Konfiguration
st.set_page_config(
    page_title="Knoten p-Färbbarkeit Rechner",
    page_icon="🔗",
    layout="wide"
)

# Vordefinierte Knoten
KNOTEN_DATENBANK = {
    "Kleeblattknoten (3_1)": {
        "N": 5,
        "points": [[1, 1], [1, 4], [2, 3], [2, 5], [3, 2], [3, 4], [4, 1], [4, 3], [5, 2], [5, 5]]
    },
    "Achterknoten (4_1)": {
        "N": 6,
        "points": [[1,1],[1,3],[2,2],[2,4],[3,3],[3,6],[4,1],[4,5],[5,4],[5,6],[6,2],[6,5]]
    },
    "Knoten 8_21": {
        "N": 8,
        "points": [[1,1],[1,3],[2,2],[2,5],[3,4],[3,6],[4,5],[4,8],[5,3],[5,7],[6,1],[6,6],[7,4],[7,8],[8,2],[8,7]]
    }
}

# Referenztabelle
TABELLE = {
    '3_1': [3], '4_1': [5], '5_1': [5], '5_2': [7], '6_1': [3],
    '6_2': [11], '6_3': [13], '7_1': [7], '7_2': [11], '7_3': [13],
    '7_4': [3, 5], '7_5': [17], '7_6': [19], '7_7': [3, 7], '8_1': [13],
    '8_2': [17], '8_3': [17], '8_4': [19], '8_5': [3, 7], '8_6': [23],
    '8_7': [23], '8_8': [5], '8_9': [5], '8_10': [3], '8_11': [3],
    '8_12': [29], '8_13': [29], '8_14': [31], '8_15': [3, 7],
    '8_16': [5, 7], '8_17': [37], '8_18': [3, 5], '8_19': [3],
    '8_20': [3], '8_21': [3, 5]
}

class KnotenAnalyse:
    def __init__(self, points, N):
        self.points = points
        self.N = N
        self.points_by_col = {}
        self.points_by_row = {}
        self.crossings = []
        self.vertical_lines = []
        self.horizontal_lines = []
        self.boegen = None
        self.num_crossings = 0
        
    def punkte_zu_kreuzungen(self):
        """Konvertiert Punkte zu Kreuzungen"""
        for col, row in self.points:
            if col not in self.points_by_col:
                self.points_by_col[col] = []
            self.points_by_col[col].append(row)
            
            if row not in self.points_by_row:
                self.points_by_row[row] = []
            self.points_by_row[row].append(col)
        
        for col in self.points_by_col:
            self.points_by_col[col].sort()
        for row in self.points_by_row:
            self.points_by_row[row].sort()
        
        # Vertikale Linien
        for col, rows in self.points_by_col.items():
            if len(rows) == 2:
                self.vertical_lines.append((col, min(rows), max(rows)))
        
        # Horizontale Linien
        for row, cols in self.points_by_row.items():
            if len(cols) == 2:
                self.horizontal_lines.append((row, min(cols), max(cols)))
        
        # Kreuzungen finden
        for v_col, v_start, v_end in self.vertical_lines:
            for h_row, h_start, h_end in self.horizontal_lines:
                if (h_start < v_col < h_end) and (v_start < h_row < v_end):
                    self.crossings.append((v_col, h_row))
        
        self.crossings.sort(key=lambda x: (x[1], x[0]))
        self.num_crossings = len(self.crossings)
        
        if self.num_crossings == 0:
            raise ValueError("Keine Kreuzungen gefunden!")
        
        self.boegen = np.zeros((self.num_crossings, self.num_crossings))
        
    def test_h_rechts(self, a, b):
        """Test für horizontale Richtung"""
        ligne = -1
        for i in range(len(self.horizontal_lines)):
            if b == self.horizontal_lines[i][0]:
                ligne = i
                break
        
        if ligne == -1:
            raise ValueError(f"Keine Zeile für {b} gefunden")
        
        return a == self.horizontal_lines[ligne][1]
    
    def test_v_oben(self, a, b):
        """Test für vertikale Richtung"""
        colonne = -1
        for i in range(len(self.vertical_lines)):
            if a == self.vertical_lines[i][0]:
                colonne = i
                break
        
        if colonne == -1:
            raise ValueError(f"Keine Spalte für {a} gefunden")
        
        return b == self.vertical_lines[colonne][2]
    
    def verti_search(self, s, a, b, v_test_zu_machen, richtung_oben=False):
        """Vertikale Suche"""
        if v_test_zu_machen:
            oben = self.test_v_oben(a, b)
        else:
            oben = richtung_oben
        
        b = b - 1 if oben else b + 1
        
        if b < 1 or b > self.N:
            raise ValueError("Vertikale Suche außerhalb der Grenzen")
        
        target = [a, b]
        
        for point in self.points:
            if target == point:
                return self.hori_search(s, a, b, True)
        
        for n, crossing in enumerate(self.crossings):
            if target == list(crossing):
                self.boegen[n][s] = 2
                break
        
        return self.verti_search(s, a, b, False, oben)
    
    def hori_search(self, s, a, b, h_test_zu_machen, richtung_rechts=True):
        """Horizontale Suche"""
        if h_test_zu_machen:
            rechts = self.test_h_rechts(a, b)
        else:
            rechts = richtung_rechts
        
        for _ in range(self.N):
            a = a + 1 if rechts else a - 1
            
            if a < 1 or a > self.N + 1:
                raise ValueError("Horizontale Suche außerhalb der Grenzen")
            
            target = [a, b]
            
            for n, crossing in enumerate(self.crossings):
                if target == list(crossing):
                    self.boegen[n][s] = -1
                    return a, b, rechts
            
            for point in self.points:
                if target == point:
                    return self.verti_search(s, a, b, True)
        
        raise ValueError("Horizontale Suche fand kein Ende")
    
    def feed_matrix(self):
        """Füllt die p-Färbbarkeitsmatrix"""
        k = 0
        a = self.crossings[k][0]
        b = self.crossings[k][1]
        richtung_rechts = True
        
        for s in range(self.num_crossings):
            crossing_found = False
            for i, crossing in enumerate(self.crossings):
                if crossing[0] == a and crossing[1] == b:
                    k = i
                    crossing_found = True
                    break
            
            if not crossing_found:
                raise ValueError(f"Kreuzung ({a},{b}) nicht gefunden")
            
            self.boegen[k][s] = -1
            
            result = self.hori_search(s, a, b, s == 0, richtung_rechts)
            if result:
                a, b, richtung_rechts = result
    
    def berechne_p_faerbbarkeit(self):
        """Berechnet die p-Färbbarkeit"""
        boegen_gestrichen = self.boegen[:-1, :-1]
        determinante = int(sp.Matrix(boegen_gestrichen).det())
        determinante_abs = abs(determinante)
        p = list(primefactors(determinante_abs))
        
        return boegen_gestrichen, determinante, p
    
    def vergleich_tabelle(self, p):
        """Vergleicht mit der Referenztabelle"""
        p_set = set(p)
        matches = [nom for nom, valeurs in TABELLE.items() if set(valeurs) == p_set]
        return matches

def zeichne_gitterdiagramm(points, N):
    """Zeichnet das Gitterdiagramm"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Grid
    for i in range(N + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5)
        ax.axvline(x=i, color='lightgray', linewidth=0.5)
    
    # Punkte
    for point in points:
        ax.plot(point[0] - 0.5, N - point[1] + 0.5, 'ro', markersize=15)
    
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(N + 1))
    ax.set_yticks(range(N + 1))
    ax.set_xticklabels([str(i) for i in range(1, N + 2)])
    ax.set_yticklabels([str(N - i + 1) for i in range(N + 1)])
    ax.grid(True, alpha=0.3)
    ax.set_title('Gitterdiagramm des Knotens', fontsize=14, fontweight='bold')
    
    return fig

def generiere_pdf(points, N, boegen_gestrichen, determinante, p, matches):
    """Generiert PDF-Report wie im Original-Code"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.cadetblue,
        spaceAfter=40,
        alignment=1
    )
    
    timestamp = datetime.datetime.now().strftime("%d/%m/%Y")
    timestamp2 = datetime.datetime.now().strftime("%H:%M:%S")
    
    story.append(Paragraph("Resultate der Knotenanalyse", title_style))
    story.append(Paragraph(f"Datum : {timestamp}", styles['Normal']))
    story.append(Paragraph(f"Zeit : {timestamp2}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Punkte
    story.append(Paragraph("Punkte des Gitterdiagramms :", styles['Heading2']))
    points_text = ", ".join([f"({int(x)}, {int(y)})" for x, y in points])
    story.append(Paragraph(points_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Gitterdiagramm als Tabelle
    story.append(Paragraph("Gitterdiagramm :", styles['Heading2']))
    diagramm = np.full((N, N), ' ', dtype=str)
    for point in points:
        diagramm[point[1]-1, point[0]-1] = '*'
    
    labeled_diagramm = []
    header_row = [''] + [str(i + 1) for i in range(N)]
    labeled_diagramm.append(header_row)
    
    for i, row in enumerate(diagramm.tolist()):
        labeled_row = [str(i + 1)] + row
        labeled_diagramm.append(labeled_row)
    
    grid_table = Table(labeled_diagramm)
    grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.8, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.cadetblue),
        ('BACKGROUND', (0, 0), (0, -1), colors.cadetblue),
        ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Courier-Bold'),
    ]))
    story.append(grid_table)
    story.append(Spacer(1, 20))
    
    # Matrix
    story.append(Paragraph("p-Färbbarkeitsmatrix :", styles['Heading2']))
    matrix_data = [[int(val) for val in row] for row in boegen_gestrichen.tolist()]
    matrix_table = Table(matrix_data, colWidths=[0.4*inch]*len(boegen_gestrichen[0]))
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0,0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0,0), (-1, -1), 1, colors.black),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.white),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.white)
    ]))
    
    display_table = Table([["A =", matrix_table]], colWidths=[0.8*inch, None])
    display_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (0, 0), 12),
        ('RIGHTPADDING', (0, 0), (0, 0), 12),
    ]))
    story.append(display_table)
    story.append(Spacer(1, 20))
    
    # Determinante
    story.append(Paragraph("Determinante der p-Färbbarkeitsmatrix :", styles['Heading2']))
    story.append(Paragraph(f"det(A) = {determinante}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # p-Werte
    story.append(Paragraph("Primzahlen p, die eine p-Färbung des Knotens erlauben :", styles['Heading2']))
    story.append(Paragraph(f"p = {p}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Erklärung und Matches
    story.append(Paragraph("Mögliche Übereinstimmung:", styles['Heading2']))
    story.append(Paragraph("Das Programm schlägt Knoten vor, "
                           "die entsprechend ihrer p-Färbbarkeit mit dem eingegebenen "
                           "Knoten übereinstimmen könnten. Die p-Färbbarkeit ist eine Invariante, "
                           "das heisst derselbe Knoten ist immer für diesselbe Primzahl p färbbar, "
                           "unabhängig von seiner Projektion. Dieser Vorschlag ist jedoch keineswegs "
                           "eine Gewissheit, da es keine absolute Invariante gibt. Um eine grössere "
                           "Sicherheit hinsichtlich des eingegebenen Knotens zu erhalten, müssten weitere "
                           "Invarianten berechnet werden. Zum Vergleich verwendet das Programm eine Tabelle"
                           "mit den p-Färbbarkeiten aller Knoten bis zu 8 Kreuzungen. Diese Tabelle wurde "
                           "erstellt, indem man sich zunutze machte, dass das Alexander-Polynom von -1 "
                           "eines Knotens gleich der Determinante der p-Färbbarkeitsmatrix ist.", styles['Normal']))
    
    if matches:
        matched_heading_style = ParagraphStyle('MatchedKnots',
                                               parent=styles['Heading3'],
                                               fontSize=10,
                                               spaceAfter=5,
                                               spaceBefore=10)
        story.append(Paragraph("Knoten mit übereinstimmenden p-Färbbarkeiten:", matched_heading_style))
        for match in matches:
            story.append(Paragraph(match, styles['Normal']))
    else:
        story.append(Paragraph("Der eingegebene Knoten hat eine minimale Kreuzungszahl von mehr als 8, da das oder die p nicht mit einem der Einträge in der Tabelle übereinstimmen.", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- STREAMLIT UI ---

st.title("🔗 Knoten p-Färbbarkeits-Rechner")
st.markdown("Berechne die p-Färbbarkeitsmatrix für mathematische Knoten aus Grid-Diagrammen")

# Sidebar
with st.sidebar:
    st.header("📋 Eingabe")
    eingabe_methode = st.radio(
        "Wähle eine Methode:",
        ["Vordefinierter Knoten", "Eigene Punkte eingeben"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.markdown("Dieses Tool berechnet die p-Färbbarkeit von Knoten basierend auf Grid-Diagrammen.")
    st.markdown("[Mehr über Grid Notation](https://knotinfo.org/descriptions/grid_notation.html)")

# Hauptbereich
if eingabe_methode == "Vordefinierter Knoten":
    knoten_wahl = st.selectbox(
        "Wähle einen Knoten:",
        list(KNOTEN_DATENBANK.keys())
    )
    
    knoten_data = KNOTEN_DATENBANK[knoten_wahl]
    points = knoten_data["points"]
    N = knoten_data["N"]
    
    st.success(f"✅ {knoten_wahl} ausgewählt (Grid-Größe: {N}×{N})")
    
    berechnen = st.button("🔍 Analyse starten", type="primary", use_container_width=True)

else:
    st.subheader("Eigene Punkte eingeben")
    
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Grid-Größe N:", min_value=2, max_value=15, value=5)
    with col2:
        num_points = 2 * N
        st.info(f"Anzahl Punkte: {num_points}")
    
    st.markdown("**Gib die Koordinaten ein (Spalte, Zeile):**")
    
    points = []
    cols = st.columns(4)
    
    for i in range(num_points):
        with cols[i % 4]:
            st.markdown(f"**Punkt {i+1}**")
            x = st.number_input(f"Spalte", min_value=1, max_value=N, value=min(i+1, N), key=f"x_{i}", label_visibility="collapsed")
            y = st.number_input(f"Zeile", min_value=1, max_value=N, value=min(i+1, N), key=f"y_{i}", label_visibility="collapsed")
            points.append([x, y])
    
    berechnen = st.button("🔍 Analyse starten", type="primary", use_container_width=True)

# Berechnung
if berechnen:
    try:
        with st.spinner("Berechnung läuft..."):
            # Analyse durchführen
            analyse = KnotenAnalyse(points, N)
            analyse.punkte_zu_kreuzungen()
            analyse.feed_matrix()
            boegen_gestrichen, determinante, p = analyse.berechne_p_faerbbarkeit()
            matches = analyse.vergleich_tabelle(p)
        
        st.success("✅ Analyse erfolgreich abgeschlossen!")
        
        # Ergebnisse anzeigen
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Gitterdiagramm")
            fig = zeichne_gitterdiagramm(points, N)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("🔢 p-Färbbarkeitsmatrix")
            st.dataframe(boegen_gestrichen, use_container_width=True)
            
            st.markdown("---")
            st.metric("Anzahl Kreuzungen", analyse.num_crossings)
            st.metric("Determinante", determinante)
            st.metric("p-Werte", ", ".join(map(str, p)))
        
        # Vergleich
        st.subheader("🔍 Vergleich mit Datenbank")
        if matches:
            st.success(f"**Mögliche Übereinstimmungen:** {', '.join(matches)}")
        else:
            st.info("Keine Übereinstimmung mit Knoten bis 8 Kreuzungen gefunden.")
        
        # PDF Download
        st.markdown("---")
        st.subheader("📥 Export")
        pdf_buffer = generiere_pdf(points, N, boegen_gestrichen, determinante, p, matches)
        
        st.download_button(
            label="📄 PDF-Report herunterladen",
            data=pdf_buffer,
            file_name=f"knoten_analyse_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Fehler bei der Berechnung: {str(e)}")
        st.exception(e)
        
        
        
