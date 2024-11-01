import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

import {
  applyIndirectStyles,
  applyTransform,
  calculateScaleFactors,
  invertPoint
} from "./style";

/**
 * A custom mark for mouse interaction on plots.
 * @extends Plot.Mark
 */
export class EventHandler extends Plot.Mark {
  /**
   * Creates a new event handler mark.
   * @param {Object} options - Configuration options for the event handler mark.
   * @param {Function} [options.onDrawStart] - Callback function called when drawing starts. Receives an event object with {type: "drawstart", x, y, startTime}.
   * @param {Function} [options.onDraw] - Callback function called during drawing. Receives an event object with {type: "draw", x, y, startTime}.
   * @param {Function} [options.onDrawEnd] - Callback function called when drawing ends. Receives an event object with {type: "drawend", x, y, startTime}.
   * @param {Function} [options.onMouseMove] - Callback function called when the mouse moves over the drawing area. Receives an event object with {type: "mousemove", x, y}.
   * @param {Function} [options.onClick] - Callback function called when the drawing area is clicked. Receives an event object with {type: "click", x, y}.
   * @param {Function} [options.onMouseDown] - Callback function called when the mouse button is pressed down. Receives an event object with {type: "mousedown", x, y, startTime}.
   */
  constructor(options = {}) {
    super([null], {}, options, {
      ariaLabel: "draw area",
      fill: "none",
      stroke: "none",
      strokeWidth: 1,
      pointerEvents: "all"
    });

    this.onDrawStart = options.onDrawStart;
    this.onDraw = options.onDraw;
    this.onDrawEnd = options.onDrawEnd;
    this.onMouseMove = options.onMouseMove;
    this.onClick = options.onClick;
    this.onMouseDown = options.onMouseDown;
  }

  /**
   * Renders the event handler mark.
   * @param {number} index - The index of the mark.
   * @param {Object} scales - The scales for the plot.
   * @param {Object} channels - The channels for the plot.
   * @param {Object} dimensions - The dimensions of the plot.
   * @param {Object} context - The rendering context.
   * @returns {SVGGElement} The rendered SVG group element.
   */
  render(index, scales, channels, dimensions, context) {
    const { width, height } = dimensions;
    let currentDrawingRect = null;
    let drawingArea = null;
    let scaleFactors;
    let drawStartTime;

    const eventData = (eventType, point) => ({
      type: eventType,
      x: point[0],
      y: point[1],
      startTime: drawStartTime
    });

    const isWithinDrawingArea = (rect, x, y) =>
      x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;

    const handleDrawStart = (point) => {
      this.onDrawStart?.(eventData("drawstart", point));
      document.addEventListener('mousemove', handleDraw);
      document.addEventListener('mouseup', handleDrawEnd);
    }

    const handleMouseDown = (event) => {
      currentDrawingRect = drawingArea.getBoundingClientRect();
      if (!isWithinDrawingArea(currentDrawingRect, event.clientX, event.clientY)) return;

      drawStartTime = Date.now();
      scaleFactors = calculateScaleFactors(drawingArea.ownerSVGElement);
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      const point = invertPoint(offsetX, offsetY, scales, scaleFactors);
      this.onMouseDown?.(eventData("mousedown", point));

      handleDrawStart(point)

    };

    const handleDraw = (event) => {
      if (!currentDrawingRect) return;
      event.preventDefault();
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      const point = invertPoint(offsetX, offsetY, scales, scaleFactors);
      this.onDraw?.(eventData("draw", point));
    };

    const handleDrawEnd = (event) => {
      if (!currentDrawingRect) return;
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      const point = invertPoint(offsetX, offsetY, scales, scaleFactors);
      this.onDrawEnd?.(eventData("drawend", point));

      document.removeEventListener('mousemove', handleDraw);
      document.removeEventListener('mouseup', handleDrawEnd);
      currentDrawingRect = null;
      drawStartTime = null;
    };

    const handleMouseMove = (event) => {
      if (this.onMouseMove) {
        const rect = drawingArea.getBoundingClientRect();
        const offsetX = event.clientX - rect.left;
        const offsetY = event.clientY - rect.top;
        const point = invertPoint(offsetX, offsetY, scales, calculateScaleFactors(drawingArea.ownerSVGElement));
        this.onMouseMove(eventData("mousemove", point));
      }
    };

    const handleClick = (event) => {
      const rect = drawingArea.getBoundingClientRect();
      if (!isWithinDrawingArea(rect, event.clientX, event.clientY)) return;

      if (this.onClick) {
        const offsetX = event.clientX - rect.left;
        const offsetY = event.clientY - rect.top;
        const point = invertPoint(offsetX, offsetY, scales, calculateScaleFactors(drawingArea.ownerSVGElement));
        this.onClick(eventData("click", point));
      }
    };

    const g = d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0);

    drawingArea = g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .on("mousemove", handleMouseMove)
      .node();

    // We attach mousedown and click to document to allow interaction even when the cursor is over other elements
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('click', handleClick);

    return g.node();
  }
}

/**
 * Returns a new event handler mark for the given options.
 * @param {Object} _data - Unused parameter (maintained for consistency with other mark functions).
 * @param {EventHandlerOptions} options - Options for the event handler mark.
 * @returns {EventHandler} A new event handler mark.
 */
export function events(_data, options = {}) {
  return new EventHandler(options);
}

/**
 * @typedef {Object} EventHandlerOptions
 * @property {Function} [onDrawStart] - Callback function called when drawing starts. Receives an event object with {type: "drawstart", x, y, startTime}.
 * @property {Function} [onDraw] - Callback function called during drawing. Receives an event object with {type: "draw", x, y, startTime}.
 * @property {Function} [onDrawEnd] - Callback function called when drawing ends. Receives an event object with {type: "drawend", x, y, startTime}.
 * @property {Function} [onMouseMove] - Callback function called when the mouse moves over the drawing area. Receives an event object with {type: "mousemove", x, y}.
 * @property {Function} [onClick] - Callback function called when the drawing area is clicked. Receives an event object with {type: "click", x, y}.
 * @property {Function} [onMouseDown] - Callback function called when the mouse button is pressed down. Receives an event object with {type: "mousedown", x, y, startTime}.
 */
