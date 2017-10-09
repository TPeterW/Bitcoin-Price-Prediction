const fs = require('fs');
var csvWriter = require('csv-write-stream')
const googleTrends = require('google-trends-api');

let start_date = new Date();
let end_date = new Date();
start_date.setHours(0);
start_date.setMinutes(0);
start_date.setSeconds(0);
start_date.setMilliseconds(0);
end_date.setHours(0);
end_date.setMinutes(0);
end_date.setSeconds(0);
end_date.setMilliseconds(0);

let oneDay = 1000 * 60 * 60 * 24;
let thirtyDays = 1000 * 60 * 60 * 24 * 30;
start_date.setFullYear(2014, 0, 1);		// month (0~11)
end_date.setTime(start_date.getTime() + thirtyDays)

while (end_date < new Date()) {
	console.log('Working on ' + start_date);
	googleTrends.interestOverTime({
		keyword: 'bitcoin',
		startTime: start_date,
		endTime: end_date,
	}).then(function(res) {
		console.log(res);
		// formatted = JSON.parse(res);
		// let item = formatted.default.timelineData[0];
		// let writer;
		// if (!fs.existsSync('data.csv')) {
		// 	writer = csvWriter({ headers: ['date', 'value'] });
		// 	writer.pipe(fs.createWriteStream('data.csv'));
		// 	writer.write([item.formattedAxisTime, item.value[0]]);
		// } else {
		// 	writer = csvWriter({ sendHeaders: false });
		// 	writer.pipe(fs.createWriteStream('data.csv', { flags: 'a' }));
		// 	writer.write({ 'date': item.formattedAxisTime, 'value': item.value[0] });
		// }
		// writer.end();
	}).catch(function(err) {
		console.log(err);
	});
	break;
	start_date.setTime(start_date.getTime() + oneDay);
	end_date.setTime(start_date.getTime() + thirtyDays);
}