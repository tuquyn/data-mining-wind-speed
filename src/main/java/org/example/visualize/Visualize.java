package org.example.visualize;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import org.example.model.StoreResult;
import org.example.model.StoreResultList;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;
import java.io.FileReader;
import java.io.IOException;
import java.sql.ResultSet;
import java.util.List;

public class Visualize {
    public static void generateCharts() {
        String filename = "result/result.json";
        ObjectMapper objectMapper = new ObjectMapper();
        Gson gson = new Gson();

        try (FileReader reader = new FileReader(filename)) {
            List<StoreResult> resultList = objectMapper.readValue(reader, new TypeReference<List<StoreResult>>(){});

            DefaultCategoryDataset mseDataset = new DefaultCategoryDataset();
            DefaultCategoryDataset maeDataset = new DefaultCategoryDataset();

            for (StoreResult storeResult : resultList) {
                for (int i = 0; i < storeResult.getMseAccuracy().size(); i++) {
                    mseDataset.addValue(storeResult.getMseAccuracy().get(i), storeResult.getModelName(), String.valueOf(i));
                    maeDataset.addValue(storeResult.getMaeAccuracy().get(i), storeResult.getModelName(), String.valueOf(i));
                }
            }

            JFreeChart mseChart = ChartFactory.createLineChart(
                    "MSE Accuracy", // Chart title
                    "Epoch", // X-axis label
                    "MSE", // Y-axis label
                    mseDataset); // data

            JFreeChart maeChart = ChartFactory.createLineChart(
                    "MAE Accuracy", // Chart title
                    "Epoch", // X-axis label
                    "MAE", // Y-axis label
                    maeDataset); // data

            // Display the charts
            ChartPanel mseChartPanel = new ChartPanel(mseChart);
            JFrame mseFrame = new JFrame("MSE Chart");
            mseFrame.setSize(800, 600);
            mseFrame.setContentPane(mseChartPanel);
            mseFrame.setLocationRelativeTo(null);
            mseFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            mseFrame.setVisible(true);

            ChartPanel maeChartPanel = new ChartPanel(maeChart);
            JFrame maeFrame = new JFrame("MAE Chart");
            maeFrame.setSize(800, 600);
            maeFrame.setContentPane(maeChartPanel);
            maeFrame.setLocationRelativeTo(null);
            maeFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            maeFrame.setVisible(true);
        } catch (IOException e) {
            System.out.println("An error occurred while reading the file.");
            e.printStackTrace();
        }
    }
}
