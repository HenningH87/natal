"""
This module provides the Chart class for generating SVG representations of natal charts.
It includes functionality for creating sign wheels, house wheels, body placements,
and aspect lines for both single and composite charts.
"""

from functools import cached_property
from math import cos, radians, sin, degrees, atan2
from natal.classes import Aspect
from natal.config import DotDict, HouseSys
from natal.const import SIGN_MEMBERS, VERTEX_NAMES
from natal.data import Data
from pathlib import Path
from natal.utils import generate_alphabetical_sequence
from tagit import circle, g, line, path, svg, text


def _svg_paths() -> dict:
    folder = Path(__file__).parent.absolute() / "svg_paths"
    return {svg.stem: svg.read_text() for svg in folder.glob("*.svg")}


svg_paths = _svg_paths()


class Chart(DotDict):
    """SVG representation of a natal chart.

    This class generates the visual components of an astrological chart,
    including sign wheels, house wheels, planet placements, and aspect lines.
    It supports both single and composite charts.
    """

    def __init__(
        self,
        data1: Data,
        width: int,
        height: int | None = None,
        data2: Data | None = None,
    ) -> None:
        """Initialize a Chart object.

        Args:
            data1: Primary chart data
            width: Width of the SVG
            height: Height of the SVG. If None, set to width
            data2: Secondary chart data for composite charts

        Returns:
            None
        """
        self.data1 = data1
        self.data2 = data2
        self.width = width
        self.height = height
        if self.height is None:
            self.height = self.width
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.config = self.data1.config
        margin = min(self.width, self.height) * self.config.chart.margin_factor
        self.max_radius = min(self.width - margin, self.height - margin) // 2
        self.margin = margin
        self.ring_thickness = (
            self.max_radius * self.config.chart.ring_thickness_fraction
        )
        self.font_size = self.ring_thickness * self.config.chart.font_size_fraction
        self.scale_adjustment = self.width / self.config.chart.scale_adj_factor
        self.pos_adjustment = self.font_size / self.config.chart.pos_adj_factor

    def svg_root(self, content: str | list[str]) -> str:
        """Generate an SVG root element with sensible defaults.

        Args:
            content: The content to be included in the SVG root

        Returns:
            An SVG root element as a string
        """
        return svg(
            content,
            height=self.height,
            width=self.width,
            font_family=self.config.chart.font,
            version="1.1",
            xmlns="http://www.w3.org/2000/svg",
        )

    def sector(
        self,
        radius: int,
        start_deg: float,
        end_deg: float,
        fill: str = "white",
        stroke_color: str = "black",
        stroke_width: float = 1,
        stroke_opacity: float = 1,
    ) -> str:
        """Create a sector shape in SVG format.

        Args:
            radius: Radius of the sector
            start_deg: Starting angle in degrees
            end_deg: Ending angle in degrees
            fill: Fill color of the sector
            stroke_color: Stroke color of the sector
            stroke_width: Width of the stroke
            stroke_opacity: Opacity of the stroke

        Returns:
            An SVG path element representing the sector
        """
        start_rad = radians(start_deg)
        end_rad = radians(end_deg)
        start_x = self.cx - radius * cos(start_rad)
        start_y = self.cy + radius * sin(start_rad)
        end_x = self.cx - radius * cos(end_rad)
        end_y = self.cy + radius * sin(end_rad)

        start_x, start_y, end_x, end_y = [
            round(val, 2) for val in (start_x, start_y, end_x, end_y)
        ]

        path_data = " ".join(
            (
                "M{} {}".format(self.cx, self.cy),
                "L{} {}".format(start_x, start_y),
                "A{} {} 0 0 0 {} {}".format(radius, radius, end_x, end_y),
                "Z",
            )
        )
        return path(
            "",
            d=path_data,
            fill=fill,
            stroke=stroke_color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )

    def background(self, radius: float, **kwargs) -> str:
        """Create a background circle for the chart.

        Args:
            radius: Radius of the background circle
            **kwargs: Additional attributes for the circle element

        Returns:
            An SVG circle element representing the background
        """
        return circle(cx=self.cx, cy=self.cy, r=radius, **kwargs)

    def sign_wheel(self) -> list[str]:
        """Generate the zodiac sign wheel.

        Returns:
            A list of SVG elements representing the sign wheel
        """
        radius = self.max_radius

        wheel = [self.background(radius=radius, fill=self.config.theme.background)]
        for i in range(12):
            start_deg = self.data1.signs[i].normalized_degree
            end_deg = start_deg + 30
            wheel.append(
                self.sector(
                    radius=radius,
                    start_deg=start_deg,
                    end_deg=end_deg,
                    fill=self.config.theme.signWheel,
                    stroke_color=self.config.theme.foreground,
                    stroke_width=self.config.chart.stroke_width,
                )
            )
        return wheel

    def sign_wheel_symbols(self) -> list[str]:
        """Generate the zodiac sign symbols for the sign wheel.

        Returns:
            A list of SVG elements representing the zodiac sign symbols
        """

        wheel = []
        for i in range(12):
            start_deg = self.data1.signs[i].normalized_degree
            symbol_radius = self.max_radius - (self.ring_thickness / 4)
            symbol_angle = radians(start_deg + 15)  # Center of the sector
            symbol_x = self.cx - symbol_radius * cos(symbol_angle) - self.pos_adjustment
            symbol_y = self.cy + symbol_radius * sin(symbol_angle) - self.pos_adjustment
            wheel.append(
                g(
                    [
                        circle(
                            cx=10,
                            cy=10,
                            r=12,
                            stroke="none",
                            fill="none",  # Sign background color
                        ),
                        svg_paths[SIGN_MEMBERS[i].name],
                    ],
                    stroke=self.config.theme.labels,
                    stroke_width=self.config.chart.stroke_width * 1.5,
                    fill="none",
                    transform=f"translate({symbol_x}, {symbol_y}) scale({self.scale_adjustment})",
                )
            )
        return wheel

    def house_wheel(self) -> list[str]:
        """Generate the house wheel with correctly rotated house numbers.

        Returns:
            A list of SVG elements representing the house wheel.
        """
        radius = self.max_radius - self.ring_thickness / 2
        wheel = [self.background(radius, fill=self.config.theme.background)]

        for i, (start_deg, end_deg) in enumerate(self.house_vertices):
            # Draw house sector
            wheel.append(
                self.sector(
                    radius=radius,
                    start_deg=start_deg,
                    end_deg=end_deg,
                    fill=self.config.theme.houses,  # House fill color
                    stroke_color=self.config.theme.foreground,
                    stroke_width=self.config.chart.stroke_width,
                )
            )

            # Add house number
            number_radius = radius - (self.ring_thickness / 4)
            mid_deg = (start_deg + end_deg) / 2 % 360  # Midpoint of the house sector
            mid_rad = radians(mid_deg)  # Convert to radians for calculations

            # Position of the house number
            number_x = self.cx - number_radius * cos(mid_rad)
            number_y = self.cy + number_radius * sin(mid_rad)

            # Calculate the angle directly from the coordinates
            relative_x = number_x - self.cx
            relative_y = number_y - self.cy
            rotation_angle = degrees(atan2(relative_y, relative_x)) + 90

            # Flip text if in the bottom half of the chart
            if 90 < rotation_angle < 270:
                rotation_angle += 180

            wheel.append(
                text(
                    str(i + 1),
                    x=number_x,
                    y=number_y,
                    fill=self.config.theme.labels,  # Label color
                    font_size=self.font_size * 0.4,
                    text_anchor="middle",
                    dominant_baseline="central",
                    transform=f"rotate({rotation_angle}, {number_x}, {number_y})",
                )
            )

        return wheel

    def vertex_wheel(self) -> list[str]:
        """Generate vertex lines for the chart.

        Returns:
            A list of SVG elements representing vertex lines
        """
        vertex_radius = self.max_radius + self.margin // 2
        house_radius = self.max_radius - 1 * self.ring_thickness
        body_radius = self.max_radius - 3 * self.ring_thickness

        lines = [
            self.background(
                house_radius,
                fill=self.config.theme.background,
                stroke=self.config.theme.foreground,
                stroke_width=self.config.chart.stroke_width,
            ),
            self.background(
                body_radius,
                fill="#88888800",  # transparent
                stroke=self.config.theme.labels,
                stroke_width=self.config.chart.stroke_width,
            ),
        ]
        for house in self.data1.houses:
            radius = house_radius
            stroke_width = self.config.chart.stroke_width
            stroke_color = self.config.theme.labels

            angle = radians(house.normalized_degree)
            end_x = self.cx - radius * cos(angle)
            end_y = self.cy + radius * sin(angle)

            lines.append(
                line(
                    x1=self.cx,
                    y1=self.cy,
                    x2=end_x,
                    y2=end_y,
                    stroke=self.config.theme.foreground,
                    stroke_width=stroke_width,
                    stroke_opacity=self.config.chart.stroke_opacity,
                )
            )

        return lines

    def outer_body_wheel(self) -> list[str]:
        """Generate the outer body wheel for single or composite charts, including alphabetical labels.

        Returns:
            A list of SVG elements representing the outer body wheel.
        """
        radius = self.max_radius - 3 * self.ring_thickness
        data = self.data2 or self.data1
        body_wheel_elements = self.body_wheel(radius, data, self.config.chart.outer_min_degree, show_degree=True)

        # Add alphabetical labels for non 12 house systems
        if self.config.house_sys == HouseSys.Whole_Sign:
            label_elements = self.add_outer_labels(self.max_radius  - 5 * self.ring_thickness / 4 )#- self.ring_thickness / 2)
        else:
            label_elements = []
        return body_wheel_elements + label_elements


    def inner_body_wheel(self) -> list[str] | None:
        """Generate the inner body wheel for composite charts.

        Returns:
            A list of SVG elements representing the inner body wheel, or None for single charts
        """
        if self.data2 is None:
            return
        radius = self.max_radius - 4 * self.ring_thickness
        data = self.data1
        return self.body_wheel(radius, data, self.config.chart.inner_min_degree)

    def outer_aspect(self) -> list[str]:
        """Generate aspect lines for the outer wheel in single charts.

        Returns:
            A list of SVG elements representing aspect lines
        """
        if self.data2 is not None:
            return []
        radius = self.max_radius - 3 * self.ring_thickness
        aspects = self.data1.aspects
        return self.aspect_lines(radius, aspects)

    def inner_aspect(self) -> list[str]:
        """Generate aspect lines for the inner wheel in composite charts.

        Returns:
            A list of SVG elements representing aspect lines
        """
        if self.data2 is None:
            return []
        radius = self.max_radius - 4 * self.ring_thickness
        aspects = self.data1.calculate_aspects(
            self.data1.composite_aspects_pairs(self.data2)
        )
        return self.aspect_lines(radius, aspects)
    
    def draw_outer_spikes(self) -> list[str]:
        """Generate planetary spikes for the outer wheel in single charts."""
        if self.data2 is not None:
            return []
        radius = self.max_radius - 3 * self.ring_thickness
        bodies = self.data1.aspectables  # Fetch all aspectable bodies
        return self.spike_lines(radius, bodies)

    def draw_inner_spikes(self) -> list[str]:
        """Generate planetary spikes for the inner wheel in composite charts."""
        if self.data2 is None:
            return []
        radius = self.max_radius - 4 * self.ring_thickness
        bodies = self.data1.aspectables + self.data2.aspectables
        return self.spike_lines(radius, bodies)
    
    def horizon(self) -> list[str]:
        """Generate horizon line for chart if desired.

        Returns:
            A list of one SVG element representing the horizon.
            The line is thick (4* the standard line width, but see through and faint)
        """
        if self.config.chart.horizon_line:
            y = self.cy  # Halfway down the image (center of the chart)
            x_start = 0  # Start at the left edge
            x_end = self.width  # End at the right edge

            return [
                line(
                    x1=x_start,
                    y1=y,
                    x2=x_end,
                    y2=y,
                    stroke=self.config.theme.horizon_color,  # Faint color
                    stroke_width=self.config.chart.stroke_width * 2,  # Wider line
                    stroke_opacity=0.25,  # Slightly transparent
                    stroke_dasharray="10,10",  # Long dashes (10 units dash, 10 units gap)
                )
            ]
        
        return []

    @property
    def svg(self) -> str:
        """Generate the SVG representation of the chart.

        Returns:
            str: SVG content.
        """
        return self.svg_root(
            [
                self.sign_wheel(),
                self.house_wheel(),
                self.vertex_wheel(),
                self.sign_wheel_symbols(),
                self.outer_body_wheel(),
                self.inner_body_wheel(),
                self.outer_aspect(),
                self.inner_aspect(),
                self.draw_outer_spikes(),
                self.draw_inner_spikes(),
                self.horizon(),
            ]
        )

    # utils ======================================================

    def adjusted_degrees(self, degrees: list[float], min_degree: float) -> list[float]:
        """Adjust spacing between celestial bodies to avoid overlap.

        Args:
            degrees: Sorted normalized degrees of celestial bodies
            min_degree: Minimum allowed degree separation

        Returns:
            Adjusted degrees of celestial bodies
        """
        step = min_degree + 0.1  # prevent overlap for float precision
        n = len(degrees)

        fwd_degs = degrees.copy()
        bwd_degs = degrees[::-1]

        # Forward adjustment
        changed = True
        while changed:
            changed = False
            for i in range(n):
                prev_deg = fwd_degs[-1] - 360 if i == 0 else fwd_degs[i - 1]
                delta = fwd_degs[i] - prev_deg
                diff = min(delta, 360 - delta)
                if (fwd_degs[i] < prev_deg) or (diff < min_degree):
                    fwd_degs[i] = prev_deg + step
                    changed = True

        # Backward adjustment
        changed = True
        while changed:
            changed = False
            for i in range(n):
                prev_deg = bwd_degs[-1] + 360 if i == 0 else bwd_degs[i - 1]
                delta = prev_deg - bwd_degs[i]
                diff = min(delta, 360 - delta)
                if (prev_deg < bwd_degs[i]) or (diff < min_degree):
                    bwd_degs[i] = prev_deg - step
                    changed = True

        bwd_degs.reverse()

        # average forward and backward adjustments
        avg_adj = []
        for fwd, bwd in zip(fwd_degs, bwd_degs):
            fwd %= 360
            bwd %= 360
            if abs(fwd - bwd) < 180:
                avg = (fwd + bwd) / 2
            else:
                avg = ((fwd + bwd + 360) / 2) % 360
            avg_adj.append(avg)

        return avg_adj

    def body_wheel(self, wheel_radius: float, data: Data, min_degree: float, show_degree:bool = False):
        """Generate elements for both inner and outer body wheels.

        Args:
            wheel_radius: Radius of the wheel
            data: Chart data to use
            min_degree: Minimum degree separation between bodies

        Returns:
            A list of SVG elements representing the body wheel
        """

        def norm_deg(x):
            return self.data1.normalize(x.degree)

        sorted_norm_bodies = sorted(data.aspectables, key=norm_deg)
        sorted_norm_degs = [norm_deg(b) for b in sorted_norm_bodies]

        # Calculate adjusted positions
        adj_norm_degs = (
            self.adjusted_degrees(sorted_norm_degs, min_degree)
            if len(sorted_norm_bodies) > 1
            else sorted_norm_degs
        )
        # for tests only
        self.adj_degs_len = len(adj_norm_degs)

        output = []
        for body, adj_deg in zip(sorted_norm_bodies, adj_norm_degs):
            g_opt = {
                "fill": "none",
                "stroke": self.config.theme.labels,
                "stroke_width": self.config.chart.stroke_width * 1.5,
            }

            # Special handling for asc, ic, dsc and mc
            if body.name in VERTEX_NAMES:
                g_opt["fill"] = self.config.theme.labels
                g_opt["stroke"] = "none"

            symbol_radius = wheel_radius + (self.ring_thickness / 2)

            # Use original angle for line start position
            original_angle = radians(self.data1.normalize(body.degree))
            degree_x = self.cx - wheel_radius * cos(original_angle)
            degree_y = self.cy + wheel_radius * sin(original_angle)

            # Use adjusted angle for symbol position
            adjusted_angle = radians(adj_deg)
            symbol_x = self.cx - symbol_radius * cos(adjusted_angle)
            symbol_y = self.cy + symbol_radius * sin(adjusted_angle)

            # Add line connecting to the inner circle
            inner_radius = wheel_radius - self.ring_thickness
            inner_x = self.cx - inner_radius * cos(original_angle)
            inner_y = self.cy + inner_radius * sin(original_angle)

            # Degree label rotation
            text_angle = adj_deg % 360
            rotation_angle = (text_angle + 2*(90-text_angle)) % 360  # Further rotate by 180 degrees
            if 90 < rotation_angle < 270:  # Flip text if on the bottom half of the circle
                rotation_angle += 180

            degree_label_x = self.cx - (symbol_radius + self.font_size) * cos(adjusted_angle)
            degree_label_y = self.cy + (symbol_radius + self.font_size) * sin(adjusted_angle)

            # Calculate background rectangle dimensions and position
            rect_width = self.font_size * 2.5
            rect_height = self.font_size * 1.2
            rect_x = degree_label_x - rect_width / 2
            rect_y = degree_label_y - rect_height / 2

            output.extend(
                [
                    line(
                        x1=degree_x,
                        y1=degree_y,
                        x2=symbol_x,
                        y2=symbol_y,
                        stroke=self.config.theme.labels,
                        stroke_width=self.config.chart.stroke_width / 2,
                    ),
                    circle(
                        cx=symbol_x,
                        cy=symbol_y,
                        r=self.font_size / 2,
                        fill=self.config.theme.background,
                    ),
                    line(
                        x1=degree_x,
                        y1=degree_y,
                        x2=inner_x,
                        y2=inner_y,
                        stroke=self.config.theme.labels,
                        stroke_width=self.config.chart.stroke_width / 2,
                        stroke_dasharray=self.ring_thickness / 11,
                    ),
                    g(
                        svg_paths[body.name],
                        transform=f"translate({symbol_x - self.pos_adjustment}, {symbol_y - self.pos_adjustment}) scale({self.scale_adjustment})",
                        **g_opt,
                    )
                ]
            )
            if show_degree:
                output.extend(
                    [
                        text(
                            f"{body.degree % 30:.1f}°",
                            x=degree_label_x,
                            y=degree_label_y,
                            fill=self.config.theme.labels,
                            font_size=self.font_size * 0.4,
                            text_anchor="middle",
                            dominant_baseline="central",
                            transform=f"rotate({rotation_angle}, {degree_label_x}, {degree_label_y})",
                        ),
                    ]
                )
        return output



    def aspect_lines(self, radius: float, aspects: list[Aspect]) -> list[str]:
        """Draw aspect lines and inward spikes for the chart, with special handling for conjunctions.

        Args:
            radius: Radius of the aspect wheel
            aspects: List of aspects to draw

        Returns:
            A list of SVG elements representing aspect lines and spikes
        """
        # Background circle for aspects
        bg = [
            self.background(
                radius,
                fill=self.config.theme.aspectBackground,
                stroke=self.config.theme.labels,
                stroke_width=self.config.chart.stroke_width,
            )
        ]

        aspect_lines = []

        # Define the inner circle radius
        inner_radius = radius - self.ring_thickness

        # Iterate through each aspect to draw the aspect lines
        for aspect in aspects:
            # Starting and ending angles for the aspect
            start_angle = radians(self.data1.normalize(aspect.body1.degree))
            end_angle = radians(self.data1.normalize(aspect.body2.degree))

            # Orb configuration and opacity factor
            orb_config = self.config.orb[aspect.aspect_member.name]
            if not orb_config:
                continue
            orb_fraction = 1 - aspect.orb / orb_config
            opacity_factor = (
                1 if aspect.aspect_member.name == "conjunction" else orb_fraction
            )

            # Adjust for the spike position (radius minus spike length)
            spike_length = self.ring_thickness * self.config.chart.spike_length_ratio
            spike_radius = radius - spike_length

            # Spike coordinates for body1
            spike_x1 = self.cx - spike_radius * cos(start_angle)
            spike_y1 = self.cy + spike_radius * sin(start_angle)

            # Spike coordinates for body2
            spike_x2 = self.cx - spike_radius * cos(end_angle)
            spike_y2 = self.cy + spike_radius * sin(end_angle)

            # Aspect line logic
            if aspect.aspect_member.name == "conjunction":
                # Draw a thick line for conjunction
                aspect_lines.append(
                    line(
                        x1=spike_x1,
                        y1=spike_y1,
                        x2=spike_x2,
                        y2=spike_y2,
                        stroke=self.config.theme.conjunction,
                        stroke_width=self.config.chart.stroke_width * self.config.chart.conjunction_line_multiple,
                        stroke_opacity=self.config.chart.stroke_opacity,
                    )
                )
            else:
                # Regular aspect line
                try:
                    _stroke = getattr(self.config.theme, aspect.aspect_member.name)
                except:
                    _stroke = self.config.theme.other_aspects
                aspect_lines.append(
                    line(
                        x1=spike_x1,
                        y1=spike_y1,
                        x2=spike_x2,
                        y2=spike_y2,
                        stroke=_stroke,
                        stroke_width=self.config.chart.stroke_width * self.config.chart.aspect_line_ratio,
                        stroke_opacity=self.config.chart.stroke_opacity * opacity_factor,
                    )
                )

        self.aspect_lines_len = len(aspect_lines)  # For testing only
        return bg + self.degree_lines(radius) + aspect_lines

    def spike_lines(self, radius: float, bodies: list[Aspect]) -> list[str]:
        """
        Generate radial spike lines for planetary bodies on the chart.

        Args:
            radius (float): Radius of the spike wheel.
            bodies (list[Aspect]): List of celestial bodies to add spikes for.

        Returns:
            list[str]: A list of SVG elements representing spike lines.
        """
        spikes = []

        spike_length = self.ring_thickness * self.config.chart.spike_length_ratio
        spike_radius = radius - spike_length

        for body in bodies:
            # Calculate spike angles
            angle = radians(self.data1.normalize(body.degree))
            
            # Inner and outer coordinates
            spike_inner_x = self.cx - spike_radius * cos(angle)
            spike_inner_y = self.cy + spike_radius * sin(angle)
            spike_outer_x = self.cx - radius * cos(angle)
            spike_outer_y = self.cy + radius * sin(angle)

            # Add spike line
            spikes.append(
                line(
                    x1=spike_inner_x,
                    y1=spike_inner_y,
                    x2=spike_outer_x,
                    y2=spike_outer_y,
                    stroke=self.config.theme.labels,
                    stroke_width=self.config.chart.stroke_width,
                )
            )

        return spikes

    def degree_lines(self, radius: float) -> list[str]:
        """
        Draw faint lines for every degree on the inner circle, inward-facing,
        with varying lengths for every fifth and tenth degree.

        Args:
            radius (float): Radius of the inner circle.

        Returns:
            list[str]: A list of SVG elements representing the degree lines.
        """
        degree_lines = []

        # Base configuration for faint lines
        base_line_length = self.ring_thickness * self.config.chart.spike_length_ratio
        line_color = self.config.theme.labels  # Use labels color
        line_opacity = 1  # Fully visible lines
        line_width = self.config.chart.stroke_width * 0.2  # Thin lines

        # Calculate line positions for 360 degrees
        for degree in range(360):
            angle = radians(degree)

            # Adjust line length for special degrees
            if degree % 10 == 0:
                line_length = base_line_length * 2  # Twice as long for every 10th degree
            elif degree % 5 == 0:
                line_length = base_line_length * 1.5  # 1.5x longer for every 5th degree
            else:
                line_length = base_line_length  # Default length

            # Start and end coordinates for the line (inward-facing)
            start_x = self.cx - radius * cos(angle)
            start_y = self.cy + radius * sin(angle)
            end_x = self.cx - (radius - line_length) * cos(angle)
            end_y = self.cy + (radius - line_length) * sin(angle)

            # Add the line to the list
            degree_lines.append(
                line(
                    x1=start_x,
                    y1=start_y,
                    x2=end_x,
                    y2=end_y,
                    stroke=line_color,
                    stroke_width=line_width,
                    stroke_opacity=line_opacity,
                )
            )

        return degree_lines

    def add_outer_labels(self, radius: float) -> list[str]:
        """
        Add alphabetical labels to the inside of the outer body wheel, starting from Aries (0°).

        Args:
            radius (float): The radius where the labels should be placed.

        Returns:
            list[str]: A list of SVG elements representing the labels.
        """
        labels = generate_alphabetical_sequence(12)  # Generate the alphabetical sequence
        label_elements = []

        # Starting angle (Aries segment start)
        start_deg = self.data1.signs[0].normalized_degree  # Assuming this gets Aries' start degree

        for i, label in enumerate(labels):
            # Calculate the angle for each label, starting from the Aries degree
            angle_deg = start_deg + i * (360 / len(labels))  # Divide the circle evenly
            angle_rad = radians(angle_deg + (30/2) % 360)  # Normalize to 360 degrees

            # Position of the label
            label_x = self.cx - radius * cos(angle_rad)
            label_y = self.cy + radius * sin(angle_rad)

            # Calculate the rotation angle based on position
            relative_x = label_x - self.cx
            relative_y = label_y - self.cy
            rotation_angle = degrees(atan2(relative_y, relative_x)) + 90

            # Flip text if in the bottom half of the chart
            if 90 < rotation_angle < 270:
                rotation_angle += 180

            # Add the label text element
            label_elements.append(
                text(
                    label,
                    x=label_x,
                    y=label_y,
                    fill=self.config.theme.labels,  # Label color
                    font_size=self.font_size * 0.4,  # Adjust font size
                    text_anchor="middle",
                    dominant_baseline="central",
                    transform=f"rotate({rotation_angle}, {label_x}, {label_y})",
                )
            )

        return label_elements




    @cached_property
    def house_vertices(self) -> list[tuple[float, float]]:
        """Calculate the vertices (start and end degrees) of each house.

        Returns:
            A list of tuples containing start and end degrees for each house
        """
        vertices = []
        for i in range(12):
            next_i = (i + 1) % 12
            start_deg = self.data1.houses[i].normalized_degree
            end_deg = self.data1.houses[next_i].normalized_degree
            # Handle the case where end_deg is less than start_deg (crosses 0°)
            if end_deg < start_deg:
                end_deg += 360
            vertices.append((start_deg, end_deg))

        return vertices

    @cached_property
    def bg_colors(self) -> list[str]:
        """Get the blended background colors for each house."""
        def hex_to_rgb(hex_value):
            hex_value = hex_value.lstrip("#")
            return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

        def rgb_to_hex(rgb):
            return "#" + "".join(f"{i:02x}" for i in rgb)

        trans = self.config.theme.transparency
        output = []
        for i in range(4):
            hex_color = self.config.theme.houses  # House base color
            rgb_color = hex_to_rgb(hex_color)
            rgb_bg = hex_to_rgb(self.config.theme.background)  # Background color
            # Blend the color with the background
            blended_rgb = tuple(
                int(trans * rgb_color[j] + (1 - trans) * rgb_bg[j]) for j in range(3)
            )
            output.append(rgb_to_hex(blended_rgb))

        return output * 4
