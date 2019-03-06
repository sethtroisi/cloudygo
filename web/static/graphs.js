// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function translate(x, y) {
    return 'translate(' + x + ',' + y + ')';
}

function add_labels(
    svg, margin, height, width,
    x, y1, y2,
    x_text, y_text, title_text) {
    // Add the X Axis
    svg.append('g')
        .attr('transform', translate(margin.left, height + margin.top))
        .call(d3.axisBottom(x));

    // Add the Y Axis
    if (y1) {
        svg.append('g')
            .attr('transform', translate(margin.left, margin.top))
            .call(d3.axisLeft(y1));
    }

    if (y2) {
        svg.append('g')
            .attr('transform', translate(width + margin.left, margin.top))
            .call(d3.axisRight(y2));
    }

    // Label for the X axis
    if (x_text) {
        svg.append('text')
            .attr('transform',
                translate(width / 2 + margin.left, height + margin.top + 30))
            .style('text-anchor', 'middle')
            .text(x_text);
    }

    if (y_text) {
        // Label for the Y axis
        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .text(y_text);
    }

    if (title_text) {
        // Title text
        svg.append('text')
            .attr('transform',
                translate(width / 2 + margin.left, margin.top / 2))
            .style('text-anchor', 'middle')
            .text(title_text);
    }
}


function add_weighted_average(group, data, fx, fy, x, y, stroke='#111', alpha=0.15) {
    var trailing_avg_data = [];
    for (i = 0; i < data.length; i++) {
        var m = 1;
        var t = 0;
        for (j = i; j >= 0 && m > 0.00001; j--) {
            t += (j == 0 ? 1 : alpha) * m * fy(data[j]);
            m *= 1 - alpha;
        }
        trailing_avg_data.push([fx(data[i]), t]);
    }

    var line = d3.line()
        .x((d) => x(d[0]))
        .y((d) => y(d[1]));

    group.append('path')
        .data([trailing_avg_data])
        .attr('d', line)
        .attr('stroke', stroke)
        .attr('stroke-width', '2px')
        .attr('fill', 'none');
}

function per_model_graph(
        svg, data, lines,
        include_average, include_right_axis, y_from_zero,
        title_text, x_text, y_text) {

    var rightMargin = 20 + 40 * include_right_axis;
    var margin = {top: 25, right: rightMargin, bottom: 50, left: 60};
    var width = svg.node().getBoundingClientRect().width -
        margin.left - margin.right;
    var height = svg.node().getBoundingClientRect().height -
        margin.bottom - margin.top;

    paths_group = svg.append('g')
        .attr('width', width)
        .attr('height', height - margin.top - margin.bottom)
        .attr('transform', translate(margin.left, margin.top));

        paths_group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#fff');

    var fx = function(d) { return d[0]; };

    // Line
    function add_line(data, x, y, funct, stroke) {
        var line = d3.line()
            .x((d) => x(fx(d)))
            .y((d) => y(funct(d)));

        return paths_group.append('path')
            .attr('class', 'data-line')
            .data([data])
            .attr('d', line)
            .attr('stroke', stroke);
    }

    // Scale the range of the data
    var x;
    if (data.length < 1 || data[0].length < 1 ||
        data[0][0] == undefined || data[0][0].getDate == null) {
      // Consider breaking out to different function
      x = d3.scaleLinear();
    } else {
      x = d3.scaleTime();
    }

    x.range([0, width]);
    var xd = d3.extent(data, fx);
    x.domain(xd);

    var yMin = d3.min(lines, func => d3.min(data, func));
    var yMax = d3.max(lines, func => d3.max(data, func));
    var yDiff = yMax - yMin;

    var y = d3.scaleLinear()
      .range([height, 0])
      .domain([y_from_zero ? 0 : yMin - 0.02 * yDiff, yMax + 0.02 * yDiff]);
    if (!y_from_zero) {
      add_line([[xd[0], 0], [xd[1], 0]], x, y,
          function(d) { return d[1]; }, '#000');
    }

    var colors = ['#111', '#151', '#115'];
    lines.forEach(function (f1, i) {
        add_line(data, x, y, f1, colors[i]);
    });

    // Exponential moving average.
    if (lines.length && include_average) {
        var trailing_avg_data = add_weighted_average(
            paths_group, data,
            fx, lines[0],
            x, y);
    }

    if (y) {
      add_labels(
          svg, margin, height, width,
          x, y, include_right_axis ? y : null,
          x_text, y_text, title_text);
    }

    return [add_line, x, y];
}


function per_model_slider_graph(
    model_num,
    svg, data, f1, f2,
    title_text, x_text, y_text) {

    svg.selectAll('*').remove();

    var margin = {top: 25, right: 20, bottom: 50, left: 60};
    var marginTB = margin.bottom + margin.top;
    var marginLR = margin.left + margin.right;
    var width = svg.node().getBoundingClientRect().width - marginLR;
    var height = svg.node().getBoundingClientRect().height - marginTB;

// set the ranges
    var x = d3.scaleLinear().range([0, width]);
    var y = d3.scaleLinear().range([height, 0]);

// define the line
    var valueline = d3.line()
        .x((d) => x(f1(d)))
        .y((d) => y(f2(d)));


    // Scale the range of the data
    x.domain(d3.extent(data, f1));
    y.domain([0, d3.max(data, f2)]);

    // TODO(sethtroisi): abstract this
    var d = x.domain();
    data = [[model_num, 0, d[0], 0], [model_num, 1, d[0], 0]]
        .concat(data).concat(
           [[model_num, 0, d[1], 0], [model_num, 1, d[1], 0]]);

    // Add the valueline path.
    paths_group = svg.append('g')
        .attr('width', width)
        .attr('height', height - marginTB)
        .attr('transform', translate(margin.left, margin.top));

    paths_group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#ffc1c1');

    paths_group.append('path')
        .data([data.filter(function(d) { return d[0] == model_num && d[1] == 1; })])
        .attr('id', 'black-length-path')
        .attr('d', valueline)
        .attr('stroke', '#111')
        .attr('fill', '#444');
    paths_group.append('path')
        .data([data.filter(function(d) { return d[0] == model_num && d[1] == 0; })])
        .attr('id', 'white-length-path')
        .attr('d', valueline)
        .attr('stroke', '#fff')
        .attr('fill', '#ddd');

    num_games = d3.sum(data, function(d) { return d[0] == model_num ? f2(d) : 0; });
    var num = model_num % 1000000;
    var full_title_text =
        title_text + ' of Model ' + num + ' (from ' + num_games + ' games)';

    add_labels(
        svg, margin, height, width,
        x, y, false /* y2 */,
        x_text, y_text, full_title_text);
}


function add_slider(slider, update_fn, min, max, init) {
    slider
        .attr('min', min)
        .attr('max', max)
        .attr('value', init)
        .on('input',function() { update_fn(parseInt(this.value)); });
    update_fn(init);
}


function find_and_set_slider(slider, data, f1, query) {
    if (query.length > 0) {
        var index = 0;
        for (var i = 0; i < data.length; i++) {
            if (f1(data[i]).indexOf(query) >= 0) {
                index = i;
                break;
            }
        }
        if (index == 0 && query.match(/^[1-9]\d*$/)) {
            index = parseInt(query);
        }
        if (index > 0 && index < data.length) {
            slider.attr('value', index);
            slider.dispatch('input');
        }
    }
}


function per_model_slider_graph_setup(
    svg, slider, data, f1, f2,
    title_text, x_text, y_text) {

    model_nums = d3.extent(data, (d) => d[0]);

    function update_graph(model_num) {
        per_model_slider_graph(
            model_num,
            svg, data, f1, f2,
            title_text, x_text, y_text);
    }

    add_slider(slider, update_graph, model_nums[0], model_nums[1], model_nums[1]);
}

/**
 * Used with figure 3 and eval page.
 * Params:
 *  f1 is name
 *  f2 is rating
 *  bucket_fn returns bucket
 *  model_fn returns an model number
 */
function rating_scatter_plot(
        svg, data, f1, f2, bucket_fn, model_fn,
        title_text, x_text, y_text) {
    var rightMargin = 20; // seperate in case we add a y2 axis label.
    var margin = {top: 25, right: rightMargin, bottom: 50, left: 60};
    var width = svg.node().getBoundingClientRect().width -
        margin.left - margin.right;
    var height = svg.node().getBoundingClientRect().height -
        margin.bottom - margin.top;

    graph_group = svg.append('g')
        .attr('width', width)
        .attr('height', height - margin.top - margin.bottom)
        .attr('transform', translate(margin.left, margin.top));

    graph_group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#fff');


    var tool_tip = d3.tip()
      .attr('class', 'd3-tip scatter-text')
      .offset([8, 0])
      .direction('s')
      .html((d) => 'Model ' + model_fn(d) + ' ranking: ' + Math.round(f1(d)));

    function add_dots_and_bars(group, points, x, y, funct, error_funct, colorScale, size) {
        var entering = group.selectAll('scatter-point')
            .data(points)
            .enter();

        var g = entering
            .append('g')
            .attr('transform', (d) => translate(x(model_fn(d)), y(funct(d))))
            .call(tool_tip);

        g.append('circle')
            .attr('r', size)
            .attr('fill', (d) => colorScale(funct(d)));
        g.append('circle')
            .attr('r', 6)
            .attr('fill-opacity', 0)
            .attr('x', '20px')
            .attr('y', '5px')
            .on('mouseover', function(e) {
                tool_tip.show(e);
                setTimeout(tool_tip.hide, 3000);
            });

        if (error_funct) {
            var error_shift = function(d) {
              // NOTE: y heads DOWN the screen.
              return y(0) - y(error_funct(d))
            }
            var negative_error_shift = function(d) {
              return -error_shift(d);
            }
            var visibility = function(d) {
              return Math.abs(error_shift(d)) > 10 ? "visible" : "hidden";
            };
            var add_line = function(class_name, x1, y1, x2, y2) {
              g.append('line')
                  .attr('class', class_name)
                  .attr('stroke', (d) => colorScale(funct(d)))
                  .attr('x1', x1)
                  .attr('y1', y1)
                  .attr('x2', x2)
                  .attr('y2', y2)
                  .attr('visibility', visibility);
            };

            add_line('error-line', '0', error_shift, '0', negative_error_shift);

            var x_scale = x(1) - x(0);
            add_line('', x_scale, error_shift,
                        -x_scale, error_shift);
            add_line('', x_scale, negative_error_shift,
                        -x_scale, negative_error_shift);
        }
    }

    // Scale the range of the data
    var x = d3.scaleLinear().range([0, width]);
    x.domain(d3.extent(data, model_fn));

    var y1;
    if (f1) {
        // Consider adding a little cushion to y1 if error bars are on.
        y1 = d3.scaleLinear()
            .domain(d3.extent(data, f1))
            .range([height, 0]);

        var colorScale = d3.scaleQuantile()
            .domain(data.map(f1))
            .range(['#F00', '#B00', '#222', '#2B2', '#2F2']);

        var bucket_data = d3.nest()
            .key(bucket_fn)
            .entries(data);

        var bucket_color = d3.schemeCategory10;
        var solo = bucket_data.length == 1;

        bucket_data.forEach(function(d, i) {
          var color = bucket_color[i];

          var bucket_group = graph_group
              .append('g')
              .attr('class', 'bucket-' + d.key);

          add_dots_and_bars(
              bucket_group,
              d.values,
              x, y1,
              f1, f2,
              solo ? colorScale : () => color,
              solo ? 3 : 1);

          var trailing_avg_data = add_weighted_average(
              bucket_group,
              d.values.filter(function(d) { return model_fn(d) > 10; }),
              model_fn, f1,
              x, y1,
              solo ? '#111' : color);
        });

        // Add legend.
        if (!solo) {
          var legend = graph_group.append("g")
            .attr("class", "legend")
            .attr("height", 100)
            .attr("width", 100)
            .attr('transform', 'translate(20, 10)')

          legend.selectAll('rect')
            .data(bucket_data)
            .enter()
            .append("rect")
              .attr("x", 20)
              .attr("y", (d, i) => height - i * 20 - 50)
              .attr("width", 10)
              .attr("height", 10)
              .style("fill", (d, i) => bucket_color[i]);

          legend.selectAll('text')
            .data(bucket_data)
            .enter()
            .append("text")
              .attr("x", 35)
              .attr("y", (d, i) => height - i * 20 - 40)
              .text((d) => d.key)
              .style("fill", (d, i) => bucket_color[i])
              .on("click", function(d) {
                var toggled = d.active == 0 ? 1 : 0;
                d.active = toggled;
                d3.selectAll(".bucket-" + d.key)
                  .transition()
                  .duration(500)
                  .style("opacity", toggled);
              });

          var active = 1;
          legend
            .append("text")
            .attr("x", 35)
            .attr("y", height - 20)
            .text("click to toggle all")
            .on("click", function() {
              var toggled = active == 0 ? 1 : 0;
              active = toggled;
              d3.selectAll("[class^='bucket-'")
                .transition()
                .duration(500)
                .style("opacity", toggled);
            });
        }
    }

    add_labels(
        svg, margin, height, width,
        x, y1, false /* y2 */,
        x_text, y_text, title_text);
}


function winrate_scatter_plot(
        svg, data, f1, f2, f3,
        title_text, x_text, y_text) {
    var rightMargin = 20; // seperate incase we add a y2 axis label.
    var margin = {top: 25, right: rightMargin, bottom: 50, left: 40};
    var width = svg.node().getBoundingClientRect().width -
        margin.left - margin.right;
    var height = svg.node().getBoundingClientRect().height -
        margin.bottom - margin.top;

    graph_group = svg.append('g')
        .attr('width', width)
        .attr('height', height - margin.top - margin.bottom)
        .attr('transform', translate(margin.left, margin.top));

    graph_group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#fff');

    var tool_tip = d3.tip()
      .attr('class', 'd3-tip scatter-text')
      .offset([8, 0])
      .direction('s')
      .html((d) => 'Model ' + d[0]);

    function add_dots(x, y, f1, f2, f3, colorScale) {
        var entering = graph_group.selectAll('scatter-point')
            .data(data).enter();

        var g = entering.append('g')
            .attr('transform', (d) => translate(x(f1(d)), y(f2(d))))
            .call(tool_tip);

        g.append('circle')
            .attr('r', function(d) { return (colorScale.domain()[1] - f3(d) < 50) ? 3 : 2; })
            .attr('fill', (d) => colorScale(f3(d)));
        g.append('circle')
            .attr('r', 6)
            .attr('fill-opacity', 0)
            .attr('x', '20px')
            .attr('y', '5px')
            .on('mouseover', function(e) {
                tool_tip.show(e);
                setTimeout(tool_tip.hide, 3000);
            });
    }

    // Scale the range of the data
    var x = d3.scaleLinear().range([0, width]);
    x.domain([0, 1]);

    var y = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);

    // TODO(sethtroisi): decide on new to old color scheme
    var colorScale = d3.scaleLinear()
        .domain(d3.extent(data, f3))
        .range(['#000', '#2D2'])
        .interpolate(d3.interpolateHcl);

    add_dots(x, y, f1, f2, f3, colorScale);

    var line = d3.line()
        .x((d) => x(d[0]))
        .y((d) => y(d[1]));

    var paths_group = graph_group.append('g');
    function add_segment(x0, y0, x1, y1, stroke) {
      paths_group.append('path')
          .attr('class', 'data-line')
          .data([[[x0, y0], [x1, y1]]])
          .attr('d', line)
          .attr('stroke', stroke);
    }

    add_segment(0, 0, 1, 1, '#000');
    add_segment(0, 0.5, 1, 0.5, '#999');
    add_segment(0.5, 0, 0.5, 1, '#999');

    add_labels(
        svg, margin, height, width,
        x, y, false /* y2 */,
        x_text, y_text, title_text);
}

