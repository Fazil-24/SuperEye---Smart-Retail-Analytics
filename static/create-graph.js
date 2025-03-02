/*
 * Parse the data and create a graph with the data.
 */
function parseData(createGraph) {
  Papa.parse("static/crowd_data.csv", {
    download: true,
    complete: function (results) {
      createGraph(results.data);
    },
  });
}

function createGraph(data) {
  var years = [];
  var hcount = ["Human count"];

  // Limit to the first 20 data points (excluding header row)
  var limitedData = data.slice(1, 51);

  for (var i = 0; i < limitedData.length; i++) {
    if (limitedData[i][1] !== undefined) {
      years.push(limitedData[i][0]); // Time values
      hcount.push(Math.round(limitedData[i][1])); // Human count
    }
  }

  let maxY = Math.max(...hcount); // Find the highest value
  let tickValues = Array.from({ length: maxY + 1 }, (_, i) => i);

  var chart = c3.generate({
    bindto: "#chart",
    data: {
      columns: [hcount],
    },
    axis: {
      x: {
        type: "category",
        categories: years,
        label: "Time",
        tick: {
          format: function (d) {
            return years[d];
          }, // Prevents '0' from appearing
        },
      },
      y: {
        label: "Crowd",
        min: 0, // Ensure Y-axis starts from 0
        values: tickValues, // Set only whole number ticks
        format: function (d) {
          return d;
        }, // Remove decimals
      },
    },
    zoom: {
      enabled: true,
    },
  });
}

parseData(createGraph);
